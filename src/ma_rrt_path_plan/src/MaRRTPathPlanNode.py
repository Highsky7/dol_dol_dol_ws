#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRT Path Planning with multiple remote goals.

author: Maxim Yastremsky(@MaxMagazin)
based on the work of AtsushiSakai(@Atsushi_twi)
"""

import rospy
import csv
import ma_rrt
import numpy as np
import time, math

from vehicle_msgs.msg import TrackCone, Track, Command, Waypoint, WaypointsArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from scipy.spatial import Delaunay
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool

class MaRRTPathPlanNode:
    def __init__(self):
        # ROS 파라미터로부터 웨이포인트 발행 여부 및 파일 경로 설정
        self.shouldPublishWaypoints = rospy.get_param('~publishWaypoints', True)  # 웨이포인트 발행 여부 (기본: True)
        self.shouldPublishPredefined = rospy.get_param('~publishPredefined', False)  # 사전 정의된 웨이포인트 발행 여부 (기본: False)

        # 파일 경로 설정 (ROS 파라미터로부터 가져옴)
        if rospy.has_param('~path'):
            self.path = rospy.get_param('~path')  # 경로 파일 경로
        if rospy.has_param('~filename'):
            self.filename = rospy.get_param('~filename')  # 파일 이름

        # 오도메트리 토픽 이름 설정 (기본: /odometry)
        if rospy.has_param('~odom_topic'):
            self.odometry_topic = rospy.get_param('~odom_topic')
        else:
            self.odometry_topic = "/odometry"

        # 월드 프레임 설정 (기본: velodyne)
        if rospy.has_param('~world_frame'):
            self.world_frame = rospy.get_param('~world_frame')
        else:
            self.world_frame = "velodyne"

        # 웨이포인트 발행 주기 설정 (기본: 5Hz)
        waypointsFrequency = rospy.get_param('~desiredWaypointsFrequency', 5)
        self.waypointsPublishInterval = 1.0 / waypointsFrequency  # 발행 간격 (초)
        self.lastPublishWaypointsTime = 0  # 마지막 웨이포인트 발행 시간

        # 장애물 존재 여부를 발행하는 퍼블리셔
        self.obstacleExistencePub = rospy.Publisher("/obstacle_existence", Bool, queue_size=1)
        
        # 구독자 설정
        rospy.Subscriber("/track", Track, self.mapCallback)  # /track 토픽에서 콘 데이터 수신
        rospy.Subscriber(self.odometry_topic, Odometry, self.odometryCallback)  # /odometry 토픽에서 차량 위치 및 자세 수신
        rospy.Subscriber("/rrt_target", PointStamped, self.rrtTargetCallback)  # /rrt_target 토픽에서 목표 지점 수신
        self.rrt_target = None  # 목표 지점 초기화
        rospy.Subscriber("/predicted_trajectory_endpoint", MarkerArray, self.predictedTrajectoryEndpointCallback)  # 예측 경로 끝점 수신
        rospy.Subscriber("/compressed_wall", PointCloud2, self.compressedWallCallback)  # 벽 장애물 포인트 클라우드 수신

        # 퍼블리셔 설정
        self.rrtTargetVisualPub = rospy.Publisher("/visual/rrt_target", Marker, queue_size=1)  # RRT 목표점 시각화
        self.waypointsPub = rospy.Publisher("/waypoints", WaypointsArray, queue_size=0)  # 전체 웨이포인트 발행
        self.newwaypointsPub = rospy.Publisher("/newwaypoints", WaypointsArray, queue_size=5)  # 새 웨이포인트 발행
        self.treeVisualPub = rospy.Publisher("/visual/tree_marker_array", MarkerArray, queue_size=0)  # RRT 트리 시각화
        self.bestBranchVisualPub = rospy.Publisher("/visual/best_tree_branch", Marker, queue_size=1)  # 최적 경로 시각화
        self.filteredBranchVisualPub = rospy.Publisher("/visual/filtered_tree_branch", Marker, queue_size=1)  # 필터링된 경로 시각화
        self.delaunayLinesVisualPub = rospy.Publisher("/visual/delaunay_lines", Marker, queue_size=1)  # 델로네 삼각형 에지 시각화
        self.waypointsVisualPub = rospy.Publisher("/visual/waypoints", MarkerArray, queue_size=1)  # 웨이포인트 시각화
        self.obstacleVisualPub = rospy.Publisher("/visual/obstacle_radius", MarkerArray, queue_size=1)  # 장애물 시각화

        # 장애물 리스트 초기화
        self.predictedEndpointObstacleList = []  # 예측 경로 끝점 장애물 리스트
        self.compressedWallObstacleList = []  # 벽 장애물 리스트
        
        # 차량 상태 초기화 (velodyne 프레임 기준)
        self.carPosX = 0.0  # 차량 x 좌표
        self.carPosY = 0.0  # 차량 y 좌표
        self.carPosYaw = 0.0  # 차량 요(yaw) 각도

        # 기타 상태 변수 초기화
        self.map = []  # 트랙 콘 데이터 저장
        self.savedWaypoints = []  # 저장된 웨이포인트 리스트
        self.preliminaryLoopClosure = False  # 예비 루프 클로저 플래그
        self.loopClosure = False  # 루프 클로저 플래그
        self.rrt = None  # RRT 객체
        self.filteredBestBranch = []  # 필터링된 최적 경로
        self.discardAmount = 0  # 경로 폐기 카운터

    def publishObstacleVisuals(self, obstacleList):
        # 장애물을 시각화하기 위해 MarkerArray 생성
        from visualization_msgs.msg import Marker, MarkerArray
        markerArray = MarkerArray()
        for i, (x, y, radius) in enumerate(obstacleList):
            # 각 장애물에 대해 구형 마커 생성
            marker = Marker()
            marker.header.frame_id = self.world_frame  # 월드 프레임 설정
            marker.header.stamp = rospy.Time.now()  # 현재 시간 스탬프
            marker.ns = "obstacle_radius"  # 네임스페이스
            marker.id = i  # 마커 ID
            marker.type = Marker.SPHERE  # 구형 마커
            marker.action = Marker.ADD  # 마커 추가
            marker.pose.position.x = x  # 장애물 x 좌표
            marker.pose.position.y = y  # 장애물 y 좌표
            marker.pose.position.z = 0.0  # z=0 (2D 평면)
            marker.pose.orientation.w = 1.0  # 기본 방향
            marker.scale.x = radius * 2.0  # 구의 x 지름 (반지름 * 2)
            marker.scale.y = radius * 2.0  # 구의 y 지름
            marker.scale.z = 0.1  # z는 얇게 (평면 표시)
            marker.color.a = 0.2  # 투명도
            marker.color.r = 1.0  # 빨간색
            marker.color.g = 0.65  # 주황빛
            marker.color.b = 0.0  # 파란색 없음
            marker.lifetime = rospy.Duration(0.2)  # 마커 지속 시간 (0.2초)
            markerArray.markers.append(marker)
        self.obstacleVisualPub.publish(markerArray)  # 마커 발행

    def rrtTargetCallback(self, msg):
        # /rrt_target 토픽에서 목표 지점 좌표 수신
        self.rrt_target = msg.point  # PointStamped 메시지의 point 저장

    def __del__(self):
        # 소멸자: 노드 종료 시 메시지 출력
        print('MaRRTPathPlanNode: Destructor called.')

    def odometryCallback(self, odometry):
        # /odometry 토픽에서 차량 위치 및 자세 수신
        from tf.transformations import euler_from_quaternion
        self.carPosX = 0.0  # velodyne 프레임 기준 x=0
        self.carPosY = 0.0  # velodyne 프레임 기준 y=0
        q = odometry.pose.pose.orientation  # 쿼터니언 추출
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])  # yaw 각도 계산
        self.carPosYaw = yaw  # 차량 방향 업데이트

    def mapCallback(self, track):
        # /track 토픽에서 트랙 콘 데이터 수신
        self.map = track.cones  # Track 메시지의 cones 필드 저장

    def predictedTrajectoryEndpointCallback(self, msg):
        # /predicted_trajectory_endpoint 토픽에서 예측 경로 끝점 처리
        for marker in msg.markers:
            x = marker.pose.position.x  # 마커의 x 좌표
            y = marker.pose.position.y  # 마커의 y 좌표
            self.predictedEndpointObstacleList.append((x, y, 0.6))  # (x, y, 반지름 0.6m) 튜플 추가

    def compressedWallCallback(self, msg):
        # /compressed_wall 토픽에서 벽 포인트 클라우드 처리
        self.compressedWallObstacleList = []  # 기존 리스트 초기화
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p[:3]  # x, y, z 좌표 추출 (z는 사용 안 함)
            self.compressedWallObstacleList.append((x, y, 0.2))  # (x, y, 반지름 0.2m) 튜플 추가

    def sampleTree(self):
        # RRT 트리 생성 및 웨이포인트 생성 함수
        self.coneObstacleList = []  # 콘 장애물 리스트 초기화
        self.coneObstacleList.clear()  # 중복 초기화 방지
        
        # 루프 클로저 활성화 시 저장된 웨이포인트 발행 후 종료
        if self.loopClosure and len(self.savedWaypoints) > 0:
            self.publishWaypoints()
            return

        # 차량 전방 12m 내의 콘만 선택
        frontConesDist = 12
        frontCones = self.getFrontConeObstacles(self.map, frontConesDist)

        coneObstacleSize = 0.6  # 콘 장애물 반지름 (미터)
        # 콘 데이터를 (x, y, 반지름) 튜플 리스트로 변환
        self.coneObstacleList = [(cone.x, cone.y, coneObstacleSize) for cone in frontCones]

        # 모든 장애물 리스트 병합 (콘 + 예측 끝점 + 벽)
        obstacleList = self.coneObstacleList + self.predictedEndpointObstacleList + self.compressedWallObstacleList
        
        # 장애물 존재 여부 발행
        exists = Bool()
        exists.data = (len(obstacleList) > 0)
        self.obstacleExistencePub.publish(exists)
        
        # 장애물 리스트 크기 로그 출력
        rospy.loginfo("-----")
        rospy.loginfo("coneObstacleList: %d", len(self.coneObstacleList))  # 콘 장애물 수
        rospy.loginfo("predictedEndpointObstacleList: %d", len(self.predictedEndpointObstacleList))  # 예측 끝점 장애물 수
        rospy.loginfo("compressedWallObstacleList: %d", len(self.compressedWallObstacleList))  # 벽 장애물 수
        rospy.loginfo("Total obstacles: %d", len(obstacleList))  # 총 장애물 수
        if not self.map:
            rospy.logwarn("No track data received")  # 트랙 데이터 미수신 경고
        if not self.compressedWallObstacleList:
            rospy.logwarn("No compressed wall data received")  # 벽 데이터 미수신 경고
        rospy.loginfo("-----")
        
        self.predictedEndpointObstacleList.clear()  # 예측 끝점 리스트 초기화
        self.publishObstacleVisuals(obstacleList)  # 장애물 시각화 발행

        rrtTarget = []  # RRT 목표 지점 리스트
        targetRadius = 0.01  # 목표 지점 반지름 (미터)
                
        # /rrt_target 토픽에서 목표 지점 수신 시
        if self.rrt_target is not None:
            rrtTarget.append((self.rrt_target.x, self.rrt_target.y, targetRadius))
            rospy.loginfo("/rrt_target: (%.2f, %.2f)", self.rrt_target.x, self.rrt_target.y)  # 목표 지점 로그
            # 목표 지점 시각화
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.world_frame
            marker.ns = "rrt_target"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 1.3
            marker.scale.y = 1.3
            marker.scale.z = 1.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.position.x = self.rrt_target.x
            marker.pose.position.y = self.rrt_target.y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            self.rrtTargetVisualPub.publish(marker)
        else:
            # 목표 지점이 없으면, 전방 6m 이상의 콘을 목표로 선택
            for cone in frontCones:
                coneDist = self.dist(self.carPosX, self.carPosY, cone.x, cone.y)
                if coneDist > 6:
                    rrtTarget.append((cone.x, cone.y, coneObstacleSize))
                    break

        # RRT 파라미터 설정
        start = [self.carPosX, self.carPosY, self.carPosYaw]  # 시작 위치 (차량 위치 및 방향)
        iterationNumber = 100  # 최대 반복 횟수
        planDistance = 5.4  # 최대 트리 확장 거리 (미터)
        expandDistance = 0.6  # 노드 간 이동 거리 (스텝 크기)
        expandAngle = 20  # 노드 생성 시 각도 제한 (도)

        # RRT 객체 생성 및 경로 계획 실행
        rrt = ma_rrt.RRT(start, planDistance, obstacleList=obstacleList, expandDis=expandDistance, turnAngle=expandAngle, maxIter=iterationNumber, rrtTargets=rrtTarget)
        nodeList, leafNodes = rrt.Planning()  # 트리 노드와 리프 노드 반환

        self.publishTreeVisual(nodeList, leafNodes)  # RRT 트리 시각화

        # 최적 경로 선택
        bestBranch = self.findBestBranch(leafNodes, nodeList, obstacleList, coneObstacleSize, expandDistance, planDistance)

        # 델로네 삼각형 에지 생성 및 시각화
        delaunayEdges = self.getDelaunayEdges(obstacleList)
        self.publishDelaunayEdgesVisual(delaunayEdges)
        
        # 웨이포인트 생성
        newWaypoints = []
        if delaunayEdges and bestBranch:
            filteredBestBranch = self.getFilteredBestBranch(bestBranch)  # 최적 경로 필터링
            if filteredBestBranch:
                newWaypoints = self.getWaypointsFromEdges(filteredBestBranch, delaunayEdges)  # 에지로부터 웨이포인트 생성
        elif delaunayEdges:
            rospy.loginfo("No bestBranch, skipping waypoint generation")  # 최적 경로 없으면 웨이포인트 생성 생략

        # 웨이포인트 병합 및 발행
        if newWaypoints:
            self.mergeWaypoints(newWaypoints)
        else:
            no_wp = Bool()
            no_wp.data = False
            self.obstacleExistencePub.publish(no_wp)  # 웨이포인트 없음 발행

        self.publishWaypoints(newWaypoints)  # 웨이포인트 발행

    def getDelaunayEdges(self, obstacleList):
        # 델로네 삼각형 에지 생성
        obstacles = [(obs[0], obs[1]) for obs in obstacleList]  # (x, y) 좌표만 추출
        rospy.loginfo("getDelaunayEdges: total obstacles=%d", len(obstacles))  # 장애물 수 로그
        if len(obstacles) < 3:
            rospy.loginfo("getDelaunayEdges: 장애물 개수 부족 (< 3), 삼각형 생성 생략")  # 최소 3개 점 필요
            return []
        pts = np.array(obstacles)  # numpy 배열로 변환
        try:
            tri = Delaunay(pts)  # 델로네 삼각형 생성
            delaunayEdges = []
            for simp in tri.simplices:  # 각 삼각형의 변을 순회
                for i in range(3):
                    j = (i + 1) % 3
                    x1, y1 = pts[simp[i]]  # 변의 시작점
                    x2, y2 = pts[simp[j]]  # 변의 끝점
                    edge = Edge(x1, y1, x2, y2)  # 에지 객체 생성
                    if edge not in delaunayEdges:  # 중복 에지 제외
                        delaunayEdges.append(edge)
            rospy.loginfo("getDelaunayEdges: 생성된 에지 수=%d", len(delaunayEdges))  # 생성된 에지 수 로그
            return delaunayEdges
        except Exception as e:
            rospy.logwarn("델로네 삼각형 생성 실패: %s", str(e))  # 예외 발생 시 경고
            return []

    def findBestBranch(self, leafNodes, nodeList, obstacleList, coneObstacleSize, expandDistance, planDistance):
        # 최적 경로 선택
        if not leafNodes:
            return  # 리프 노드 없으면 종료
        coneDistLimit = 4.0  # 장애물 고려 거리 제한 (미터)
        coneDistanceLimitSq = coneDistLimit * coneDistLimit  # 거리 제곱
        wallSafetyMargin = 0.35  # 벽과의 안전 거리 (미터)
        penalty_factor = 20.0  # 벽 페널티 계수
        epsilon = 0.01  # 제로 나누기 방지
        bothSidesImproveFactor = 3  # 양쪽 장애물 존재 시 점수 가중치
        minAcceptableBranchRating = 90  # 최소 허용 경로 점수
        leafRatings = []  # 리프 노드별 점수 리스트
        for leaf in leafNodes:
            branchRating = 0  # 경로 점수 초기화
            node = leaf  # 현재 리프 노드
            while node.parent is not None:
                nodeRating = 0  # 노드 점수 초기화
                leftObstacles = []  # 왼쪽 장애물 리스트
                rightObstacles = []  # 오른쪽 장애물 리스트
                for obs in obstacleList:
                    x, y, radius = obs  # 장애물 좌표 및 반지름
                    distSq = ((x - node.x) ** 2 + (y - node.y) ** 2)  # 노드와의 거리 제곱
                    if distSq < coneDistanceLimitSq:
                        actualDist = math.sqrt(distSq)
                        if actualDist < radius:
                            continue  # 장애물 반지름 내는 무시
                        nodeRating += (coneDistLimit - actualDist)  # 거리 기반 점수 추가
                        temp_cone = type('Cone', (), {'x': x, 'y': y})()  # isLeftCone 호환용 임시 객체
                        if self.isLeftCone(node, nodeList[node.parent], temp_cone):
                            leftObstacles.append(obs)
                        else:
                            rightObstacles.append(obs)
                for obs in obstacleList:
                    x, y, radius = obs
                    wallDistSq = (x - node.x) ** 2 + (y - node.y) ** 2
                    if wallDistSq < (radius + wallSafetyMargin) ** 2:
                        nodeRating -= penalty_factor / (wallDistSq + epsilon)  # 벽 근접 시 페널티
                if (len(leftObstacles) == 0 and len(rightObstacles) > 0) or (len(leftObstacles) > 0 and len(rightObstacles) == 0):
                    nodeRating /= bothSidesImproveFactor  # 한쪽만 장애물일 경우 점수 감소
                if len(leftObstacles) > 0 and len(rightObstacles) > 0:
                    nodeRating *= bothSidesImproveFactor  # 양쪽 장애물일 경우 점수 증가
                nodeFactor = (node.cost - expandDistance) / (planDistance - expandDistance) + 1  # 노드 비용 가중치
                branchRating += nodeRating * nodeFactor  # 경로 점수 누적
                node = nodeList[node.parent]  # 부모 노드로 이동
            leafRatings.append(branchRating)
        maxRating = max(leafRatings)  # 최대 경로 점수
        maxRatingInd = leafRatings.index(maxRating)  # 최대 점수 인덱스
        node = leafNodes[maxRatingInd]  # 최적 리프 노드
        if maxRating < minAcceptableBranchRating:
            return  # 점수 부족 시 종료
        self.publishBestBranchVisual(nodeList, node)  # 최적 경로 시각화
        reverseBranch = [node]  # 경로 역순 구성
        while node.parent is not None:
            node = nodeList[node.parent]
            reverseBranch.append(node)
        directBranch = [n for n in reversed(reverseBranch)]  # 순방향 경로
        return directBranch

    def isLeftCone(self, node, parentNode, cone):
        # 노드와 부모 노드 기준으로 장애물이 왼쪽에 있는지 확인
        return ((node.x - parentNode.x) * (cone.y - parentNode.y) - (node.y - parentNode.y) * (cone.x - parentNode.x)) > 0

    def dist(self, x1, y1, x2, y2, shouldSqrt=True):
        # 두 점 간 유클리드 거리 계산
        distSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return math.sqrt(distSq) if shouldSqrt else distSq

    def mergeWaypoints(self, newWaypoints):
        # 새 웨이포인트를 기존 웨이포인트와 병합
        if not newWaypoints:
            no_wp = Bool()
            no_wp.data = False
            self.obstacleExistencePub.publish(no_wp)  # 웨이포인트 없음 발행
            return
        maxDistToSaveWaypoints = 2.0  # 차량과 웨이포인트 간 최대 거리
        maxWaypointAmountToSave = 2  # 저장할 최대 웨이포인트 수
        waypointsDistTollerance = 1000  # 웨이포인트 중복 제거 기준 거리
        if len(self.savedWaypoints) > 15:
            firstSavedWaypoint = self.savedWaypoints[0]
            for waypoint in reversed(newWaypoints):
                distDiff = self.dist(firstSavedWaypoint[0], firstSavedWaypoint[1], waypoint[0], waypoint[1])
                if distDiff < waypointsDistTollerance:
                    self.preliminaryLoopClosure = False  # 루프 클로저 비활성화
                    break
        newSavedPoints = []
        for i in range(len(newWaypoints)):
            waypointCandidate = newWaypoints[i]
            carWaypointDist = self.dist(self.carPosX, self.carPosY, waypointCandidate[0], waypointCandidate[1])
            if i >= maxWaypointAmountToSave or carWaypointDist > maxDistToSaveWaypoints:
                break
            else:
                for savedWaypoint in reversed(self.savedWaypoints):
                    waypointsDistDiff = self.dist(savedWaypoint[0], savedWaypoint[1], waypointCandidate[0], waypointCandidate[1])
                    if waypointsDistDiff < waypointsDistTollerance:
                        self.savedWaypoints.remove(savedWaypoint)  # 중복 웨이포인트 제거
                        break
                if self.preliminaryLoopClosure:
                    distDiff = self.dist(firstSavedWaypoint[0], firstSavedWaypoint[1], waypointCandidate[0], waypointCandidate[1])
                    if distDiff < waypointsDistTollerance:
                        self.loopClosure = False  # 루프 클로저 비활성화
                        break
                self.savedWaypoints.append(waypointCandidate)  # 새 웨이포인트 추가
                newSavedPoints.append(waypointCandidate)
        if newSavedPoints:
            for point in newSavedPoints:
                newWaypoints.remove(point)  # 저장된 웨이포인트 제거

    def getWaypointsFromEdges(self, filteredBranch, delaunayEdges):
        # 델로네 에지와 최적 경로 교차점에서 웨이포인트 생성
        if not delaunayEdges:
            return []
        waypoints = []
        for i in range(len(filteredBranch) - 1):
            node1 = filteredBranch[i]
            node2 = filteredBranch[i + 1]
            a1 = np.array([node1.x, node1.y])  # 경로 시작점
            a2 = np.array([node2.x, node2.y])  # 경로 끝점
            maxAcceptedEdgeLength = 7  # 최대 허용 에지 길이
            maxEdgePartsRatio = 3  # 에지 분할 비율 제한
            intersectedEdges = []
            for edge in delaunayEdges:
                b1 = np.array([edge.x1, edge.y1])  # 에지 시작점
                b2 = np.array([edge.x2, edge.y2])  # 에지 끝점
                if self.getLineSegmentIntersection(a1, a2, b1, b2):  # 선분 교차 여부
                    if edge.length() < maxAcceptedEdgeLength:
                        edge.intersection = self.getLineIntersection(a1, a2, b1, b2)  # 교차점 계산
                        edgePartsRatio = edge.getPartsLengthRatio()  # 에지 분할 비율
                        if edgePartsRatio < maxEdgePartsRatio:
                            intersectedEdges.append(edge)
            if intersectedEdges:
                if len(intersectedEdges) == 1:
                    edge = intersectedEdges[0]
                    waypoints.append(edge.getMiddlePoint())  # 단일 교차 에지 중간점 추가
                else:
                    # 여러 에지 교차 시 거리순 정렬 후 중간점 추가
                    intersectedEdges.sort(key=lambda edge: self.dist(node1.x, node1.y, edge.intersection[0], edge.intersection[1], shouldSqrt=False))
                    for edge in intersectedEdges:
                        waypoints.append(edge.getMiddlePoint())
        return waypoints

    def publishWaypoints(self, newWaypoints=None):
        # 웨이포인트 발행
        if (time.time() - self.lastPublishWaypointsTime) < self.waypointsPublishInterval:
            return  # 발행 주기 미도달 시 종료
        waypointsArray = WaypointsArray()
        newwaypointsArray = WaypointsArray()
        waypointsArray.header.frame_id = self.world_frame
        waypointsArray.header.stamp = rospy.Time.now()
        newwaypointsArray.header.frame_id = self.world_frame
        newwaypointsArray.header.stamp = rospy.Time.now()
        for i in range(len(self.savedWaypoints)):
            waypoint = self.savedWaypoints[i]
            waypointId = len(waypointsArray.waypoints)
            w = Waypoint(waypoint[0], waypoint[1], waypointId)  # 저장된 웨이포인트 추가
            waypointsArray.waypoints.append(w)
        if newWaypoints is not None:
            for i in range(len(newWaypoints)):
                waypoint = newWaypoints[i]
                waypointId = len(waypointsArray.waypoints)
                w = Waypoint(waypoint[0], waypoint[1], waypointId)  # 새 웨이포인트 추가
                waypointsArray.waypoints.append(w)
                newwaypointsArray.waypoints.append(w)
        if self.shouldPublishWaypoints:
            self.waypointsPub.publish(waypointsArray)  # 전체 웨이포인트 발행
            self.newwaypointsPub.publish(newwaypointsArray)  # 새 웨이포인트 발행
            self.lastPublishWaypointsTime = time.time()
            self.publishWaypointsVisuals(newWaypoints)  # 웨이포인트 시각화

    def publishWaypointsVisuals(self, newWaypoints=None):
        # 웨이포인트 시각화
        markerArray = MarkerArray()
        savedWaypointsMarker = Marker()
        savedWaypointsMarker.header.frame_id = self.world_frame
        savedWaypointsMarker.header.stamp = rospy.Time.now()
        savedWaypointsMarker.lifetime = rospy.Duration(1)
        savedWaypointsMarker.ns = "saved-publishWaypointsVisuals"
        savedWaypointsMarker.id = 1
        savedWaypointsMarker.type = savedWaypointsMarker.SPHERE_LIST
        savedWaypointsMarker.action = savedWaypointsMarker.ADD
        savedWaypointsMarker.pose.orientation.w = 1
        savedWaypointsMarker.scale.x = 0.15
        savedWaypointsMarker.scale.y = 0.15
        savedWaypointsMarker.scale.z = 0.15
        savedWaypointsMarker.color.a = 0.1
        savedWaypointsMarker.color.r = 0.0
        savedWaypointsMarker.color.g = 0.5
        savedWaypointsMarker.color.b = 1.0
        for waypoint in self.savedWaypoints:
            p = Point(waypoint[0], waypoint[1], 0.0)  # 저장된 웨이포인트 좌표
            savedWaypointsMarker.points.append(p)
        markerArray.markers.append(savedWaypointsMarker)
        if newWaypoints is not None:
            newWaypointsMarker = Marker()
            newWaypointsMarker.header.frame_id = self.world_frame
            newWaypointsMarker.header.stamp = rospy.Time.now()
            newWaypointsMarker.lifetime = rospy.Duration(1)
            newWaypointsMarker.ns = "new-publishWaypointsVisuals"
            newWaypointsMarker.id = 2
            newWaypointsMarker.type = newWaypointsMarker.SPHERE_LIST
            newWaypointsMarker.action = newWaypointsMarker.ADD
            newWaypointsMarker.pose.orientation.w = 1
            newWaypointsMarker.scale.x = 0.25
            newWaypointsMarker.scale.y = 0.25
            newWaypointsMarker.scale.z = 0.25
            newWaypointsMarker.color.a = 1.0
            newWaypointsMarker.color.r = 0.0
            newWaypointsMarker.color.g = 1.0
            newWaypointsMarker.color.b = 1.0
            for waypoint in newWaypoints:
                p = Point(waypoint[0], waypoint[1], 0.0)  # 새 웨이포인트 좌표
                newWaypointsMarker.points.append(p)
            markerArray.markers.append(newWaypointsMarker)
        self.waypointsVisualPub.publish(markerArray)  # 웨이포인트 마커 발행

    def getLineIntersection(self, a1, a2, b1, b2):
        # 두 선분의 교차점 계산 (무한 직선 기준)
        s = np.vstack([a1, a2, b1, b2])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:
            return (float('inf'), float('inf'))  # 평행선인 경우
        return (x/z, y/z)  # 교차점 좌표

    def getLineSegmentIntersection(self, a1, a2, b1, b2):
        # 두 선분의 실제 교차 여부 확인
        return self.ccw(a1, b1, b2) != self.ccw(a2, b1, b2) and self.ccw(a1, a2, b1) != self.ccw(a1, a2, b2)

    def ccw(self, A, B, C):
        # CCW 알고리즘으로 점의 방향성 확인
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def getFilteredBestBranch(self, bestBranch):
        # 최적 경로 필터링 (노드 위치 부드럽게 조정)
        if not bestBranch:
            return
        everyPointDistChangeLimit = 2.0  # 노드 위치 변경 제한
        newPointFilter = 0.2  # 새 위치 반영 비율
        maxDiscardAmountForReset = 2  # 최대 폐기 횟수
        if not self.filteredBestBranch:
            self.filteredBestBranch = list(bestBranch)  # 초기화
        else:
            changeRate = 0
            shouldDiscard = False
            for i in range(len(bestBranch)):
                node = bestBranch[i]
                filteredNode = self.filteredBestBranch[i]
                dist = math.sqrt((node.x - filteredNode.x) ** 2 + (node.y - filteredNode.y) ** 2)
                if dist > everyPointDistChangeLimit:
                    shouldDiscard = True
                    self.discardAmount += 1
                    if self.discardAmount >= maxDiscardAmountForReset:
                        self.discardAmount = 0
                        self.filteredBestBranch = list(bestBranch)  # 리셋
                    break
                changeRate += (everyPointDistChangeLimit - dist)
            if not shouldDiscard:
                for i in range(len(bestBranch)):
                    self.filteredBestBranch[i].x = self.filteredBestBranch[i].x * (1 - newPointFilter) + newPointFilter * bestBranch[i].x
                    self.filteredBestBranch[i].y = self.filteredBestBranch[i].y * (1 - newPointFilter) + newPointFilter * bestBranch[i].y
                self.discardAmount = 0
        self.publishFilteredBranchVisual()  # 필터링된 경로 시각화
        return list(self.filteredBestBranch)

    def publishDelaunayEdgesVisual(self, edges):
        # 델로네 삼각형 에지 시각화
        if not edges:
            return
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(1)
        marker.ns = "publishDelaunayLinesVisual"
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.pose.orientation.w = 1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.lifetime = rospy.Duration(0.1)
        for edge in edges:
            p1 = Point(edge.x1, edge.y1, 0)
            p2 = Point(edge.x2, edge.y2, 0)
            marker.points.append(p1)
            marker.points.append(p2)
        self.delaunayLinesVisualPub.publish(marker)

    def publishBestBranchVisual(self, nodeList, leafNode):
        # 최적 경로 시각화
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.2)
        marker.ns = "publishBestBranchVisual"
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.pose.orientation.w = 1
        marker.color.a = 1.0
        marker.color.r = 1.0
        node = leafNode
        parentNodeInd = node.parent
        while parentNodeInd is not None:
            parentNode = nodeList[parentNodeInd]
            p = Point(node.x, node.y, 0)
            marker.points.append(p)
            p = Point(parentNode.x, parentNode.y, 0)
            marker.points.append(p)
            parentNodeInd = node.parent
            node = parentNode
        self.bestBranchVisualPub.publish(marker)

    def publishFilteredBranchVisual(self):
        # 필터링된 최적 경로 시각화
        if not self.filteredBestBranch:
            return
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.3)
        marker.ns = "publishFilteredBranchVisual"
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.7
        marker.pose.orientation.w = 1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 122.0 / 255.0
        marker.color.b = 204.0 / 255.0
        for i in range(len(self.filteredBestBranch)):
            node = self.filteredBestBranch[i]
            p = Point(node.x, node.y, 0)
            if i != 0:
                marker.points.append(p)
            if i != len(self.filteredBestBranch) - 1:
                marker.points.append(p)
        self.filteredBranchVisualPub.publish(marker)

    def publishTreeVisual(self, nodeList, leafNodes):
        # RRT 트리와 리프 노드 시각화
        if not nodeList and not leafNodes:
            return
        markerArray = MarkerArray()
        treeMarker = Marker()
        treeMarker.header.frame_id = self.world_frame
        treeMarker.header.stamp = rospy.Time.now()
        treeMarker.ns = "rrt"
        treeMarker.type = treeMarker.LINE_LIST
        treeMarker.action = treeMarker.ADD
        treeMarker.scale.x = 0.03
        treeMarker.pose.orientation.w = 1
        treeMarker.color.a = 0.5
        treeMarker.color.g = 0.7
        treeMarker.lifetime = rospy.Duration(0.2)
        for node in nodeList:
            if node.parent is not None:
                p = Point(node.x, node.y, 0)
                treeMarker.points.append(p)
                p = Point(nodeList[node.parent].x, nodeList[node.parent].y, 0)
                treeMarker.points.append(p)
        markerArray.markers.append(treeMarker)
        leavesMarker = Marker()
        leavesMarker.header.frame_id = self.world_frame
        leavesMarker.header.stamp = rospy.Time.now()
        leavesMarker.lifetime = rospy.Duration(0.2)
        leavesMarker.ns = "rrt-leaves"
        leavesMarker.type = leavesMarker.SPHERE_LIST
        leavesMarker.action = leavesMarker.ADD
        leavesMarker.pose.orientation.w = 1
        leavesMarker.scale.x = 0.15
        leavesMarker.scale.y = 0.15
        leavesMarker.scale.z = 0.15
        leavesMarker.color.a = 1.0
        leavesMarker.color.b = 0.1
        for node in leafNodes:
            p = Point(node.x, node.y, 0)
            leavesMarker.points.append(p)
        markerArray.markers.append(leavesMarker)
        self.treeVisualPub.publish(markerArray)

    def getFrontConeObstacles(self, map, frontDist):
        # 차량 전방 지정 거리 내의 콘 선택
        if not map:
            return []
        headingVector = self.getHeadingVector()  # 차량 진행 방향 벡터
        headingVectorOrt = [-headingVector[1], headingVector[0]]  # 직교 벡터
        behindDist = 1.0  # 후방 거리
        carPosBehindPoint = [self.carPosX - behindDist * headingVector[0], self.carPosY - behindDist * headingVector[1]]  # 후방 기준점
        frontDistSq = frontDist ** 2  # 전방 거리 제곱
        frontConeList = []
        for cone in map:
            # 차량 전방에 있는 콘만 선택
            if (headingVectorOrt[0] * (cone.y - carPosBehindPoint[1]) - headingVectorOrt[1] * (cone.x - carPosBehindPoint[0])) < 0:
                if ((cone.x) ** 2 + (cone.y) ** 2) < frontDistSq:
                    frontConeList.append(cone)
        return frontConeList

    def getHeadingVector(self):
        # 차량 진행 방향 벡터 계산
        headingVector = [1.0, 0]
        carRotMat = np.array([[math.cos(self.carPosYaw), -math.sin(self.carPosYaw)], [math.sin(self.carPosYaw), math.cos(self.carPosYaw)]])
        headingVector = np.dot(carRotMat, headingVector)
        return headingVector

    def getConesInRadius(self, map, x, y, radius):
        # 지정 반경 내의 콘 선택
        coneList = []
        radiusSq = radius * radius
        for cone in map:
            if ((cone.x - x) ** 2 + (cone.y - y) ** 2) < radiusSq:
                coneList.append(cone)
        return coneList

class Edge:
    def __init__(self, x1, y1, x2, y2):
        # 델로네 에지 객체 초기화
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.intersection = None  # 교차점 초기화

    def getMiddlePoint(self):
        # 에지의 중간점 계산
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def length(self):
        # 에지 길이 계산
        return math.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)

    def getPartsLengthRatio(self):
        # 교차점 기준으로 에지 분할 비율 계산
        import math
        part1Length = math.sqrt((self.x1 - self.intersection[0]) ** 2 + (self.y1 - self.intersection[1]) ** 2)
        part2Length = math.sqrt((self.intersection[0] - self.x2) ** 2 + (self.intersection[1] - self.y2) ** 2)
        return max(part1Length, part2Length) / min(part1Length, part2Length)

    def __eq__(self, other):
        # 에지 동일성 비교
        return (self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2
                or self.x1 == other.x2 and self.y1 == other.y2 and self.x2 == other.x1 and self.y2 == other.y1)

    def __str__(self):
        # 에지 문자열 표현
        return f"({round(self.x1, 2)}, {round(self.y1, 2)}), ({round(self.x2, 2)}, {round(self.y2, 2)})"

    def __repr__(self):
        return str(self)

if __name__ == '__main__':
    maNode = MaRRTPathPlanNode()  # 노드 실행