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
        self.shouldPublishWaypoints = rospy.get_param('~publishWaypoints', True)
        self.shouldPublishPredefined = rospy.get_param('~publishPredefined', False)

        if rospy.has_param('~path'):
            self.path = rospy.get_param('~path')

        if rospy.has_param('~filename'):
            self.filename = rospy.get_param('~filename')

        if rospy.has_param('~odom_topic'):
            self.odometry_topic = rospy.get_param('~odom_topic')
        else:
            self.odometry_topic = "/odometry"

        if rospy.has_param('~world_frame'):
            self.world_frame = rospy.get_param('~world_frame')
        else:
            self.world_frame = "velodyne"

        waypointsFrequency = rospy.get_param('~desiredWaypointsFrequency', 5)
        self.waypointsPublishInterval = 1.0 / waypointsFrequency
        self.lastPublishWaypointsTime = 0

        # 새로 추가: 장애물 유무 발행
        self.obstacleExistencePub = rospy.Publisher("/obstacle_existence", Bool, queue_size=1)
        
        """
        구독자들
        """
        # 기존 /track 토픽 구독
        rospy.Subscriber("/track", Track, self.mapCallback)
        
        # 새로 추가: /odometry 토픽 구독하여 차량 위치를 업데이트
        rospy.Subscriber(self.odometry_topic, Odometry, self.odometryCallback)   
        
         # /rrt_target 토픽 구독 (ROIPathPublisher에서 퍼블리시한, roi_arc_length 경로의 끝 지점 좌표)
        rospy.Subscriber("/rrt_target", PointStamped, self.rrtTargetCallback)
        self.rrt_target = None  # 수신한 목표 좌표 저장
        
        # /predicted_trajectory_endpoint 토픽 구독자 추가
        rospy.Subscriber("/predicted_trajectory_endpoint", MarkerArray, self.predictedTrajectoryEndpointCallback)

        rospy.Subscriber("/compressed_wall", PointCloud2, self.compressedWallCallback)

        """
        퍼블리셔들
        """
        # rrt 목표점 시각화를 위한 퍼블리셔 (/visual/rrt_target)
        self.rrtTargetVisualPub = rospy.Publisher("/visual/rrt_target", Marker, queue_size=1)               

        self.waypointsPub = rospy.Publisher("/waypoints", WaypointsArray, queue_size=0)
        self.newwaypointsPub = rospy.Publisher("/newwaypoints", WaypointsArray, queue_size=5)

        self.treeVisualPub = rospy.Publisher("/visual/tree_marker_array", MarkerArray, queue_size=0)
        self.bestBranchVisualPub = rospy.Publisher("/visual/best_tree_branch", Marker, queue_size=1)
        self.filteredBranchVisualPub = rospy.Publisher("/visual/filtered_tree_branch", Marker, queue_size=1)
        self.delaunayLinesVisualPub = rospy.Publisher("/visual/delaunay_lines", Marker, queue_size=1)
        self.waypointsVisualPub = rospy.Publisher("/visual/waypoints", MarkerArray, queue_size=1)
        self.obstacleVisualPub = rospy.Publisher("/visual/obstacle_radius", MarkerArray, queue_size=1)

        # sortros 노드에서 발행한 궤적 끝점을 저장할 변수
        self.predictedEndpointObstacleList = []


        self.compressedWallObstacleList = []
        
                
        # 차량의 현재 위치 및 자세 초기값 (in velodyne frame)
        self.carPosX = 0.0
        self.carPosY = 0.0
        self.carPosYaw = 0.0

        self.map = []
        self.savedWaypoints = []
        self.preliminaryLoopClosure = False
        self.loopClosure = False
        self.rrt = None
        self.filteredBestBranch = []
        self.discardAmount = 0
        
        
    def publishObstacleVisuals(self, obstacleList):
        from visualization_msgs.msg import Marker, MarkerArray
        markerArray = MarkerArray()
        for i, (x, y, radius) in enumerate(obstacleList):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacle_radius"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # SPHERE의 scale은 지름이므로, 반지름 * 2 설정
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = 0.1  # 평면 상의 표시이므로 z는 작게

            marker.color.a = 0.2  # 투명도
            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.0

            # 필요한 경우, marker.lifetime 설정 (예: 0.2초)
            marker.lifetime = rospy.Duration(0.2)
            
            markerArray.markers.append(marker)
        self.obstacleVisualPub.publish(markerArray)
    
        
        
    def rrtTargetCallback(self, msg):
        # /rrt_target 토픽으로부터 받은 좌표를 저장 (메시지는 velodyne_frame 기준)
        self.rrt_target = msg.point

    def __del__(self):
        print('MaRRTPathPlanNode: Destructor called.')

    def odometryCallback(self, odometry):
        # # /odometry 토픽으로부터 받은 Odometry 메시지를 이용하여 차량 위치 업데이트
        # # 하지만 우리는 velodyne 좌표계 상에서의 원점 (라이다가 설치된 후륜축 중심)을 차량 위치로 가정하기 때문에 0, 0으로 설정
        self.carPosX = 0.0
        self.carPosY = 0.0
        
        # # yaw 값은 quaternion에서 추출해야 함
        from tf.transformations import euler_from_quaternion
        q = odometry.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.carPosYaw = yaw

    def yawCallback(self, yaw):
        self.carPosYaw = yaw.orientation.z

    def mapCallback(self, track):
        self.map = track.cones









    def predictedTrajectoryEndpointCallback(self, msg):
        # /predicted_trajectory_endpoint 토픽의 MarkerArray 메시지를 처리하는 콜백
        # 이전에 저장된 예측 끝점 리스트를 초기화
        # self.trajectoryEndpointsList = []
        # 수신된 모든 마커를 순회하여 좌표 저장
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            # 각 마커 중심 좌표를 반지름 ~m 장애물로 추가
            self.predictedEndpointObstacleList.append((x, y, 0.6))




    def compressedWallCallback(self, msg):
        # Clear and refill the wall obstacle list on each new message
        self.compressedWallObstacleList = []
        # Read all points (x, y, z) from the PointCloud2 message
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p[:3]
            # Append as a circular obstacle with radius ~m
            self.compressedWallObstacleList.append((x, y, 0.3))





    def sampleTree(self):
        
        self.coneObstacleList = []
        self.coneObstacleList.clear()
        # self.predictedEndpointObstacleList.clear()
                
        
        if self.loopClosure and len(self.savedWaypoints) > 0:
            self.publishWaypoints()
            return

        # cone이 없어도 트리 생성을 계속하기 위해 주석처리 (06.30)
        # if not self.map:
        #     return
        
        

        frontConesDist = 12
        frontCones = self.getFrontConeObstacles(self.map, frontConesDist)

        coneObstacleSize = 0.6  # 트래픽 콘 장애물의 반지름 (0.8m)
        # 트래픽 콘들로 이루어진 장애물 리스트 생성
        self.coneObstacleList = [(cone.x, cone.y, coneObstacleSize) for cone in frontCones]

        # 두 종류의 장애물 리스트 병합
        obstacleList = self.coneObstacleList + self.predictedEndpointObstacleList + self.compressedWallObstacleList
        
        # 새로 추가: 장애물 유무 발행
        exists = Bool()
        exists.data = (len(obstacleList) > 0)
        self.obstacleExistencePub.publish(exists)
        
        
        rospy.loginfo("-----")
        rospy.loginfo("coneObstacleList: %d", len(self.coneObstacleList))
        rospy.loginfo("predictedEndpointObstacleList: %d", len(self.predictedEndpointObstacleList))
        rospy.loginfo("compressedWallObstacleList: %d", len(self.compressedWallObstacleList))
        rospy.loginfo("Total obstacles: %d", len(obstacleList))
        rospy.loginfo("-----")
        
        self.predictedEndpointObstacleList.clear()
        
        
        # 병합된 장애물들에 대한 시각화 메시지 퍼블리시
        self.publishObstacleVisuals(obstacleList)


        
        # 이후 기존 코드대로 rrtTarget 설정 및 RRT 실행
        rrtTarget = []
        targetRadius = 0.01  # 원하는 보수적인 rrt_target 반경 값
                
        if self.rrt_target is not None:
            rrtTarget.append((self.rrt_target.x, self.rrt_target.y, targetRadius))
            rospy.loginfo("/rrt_target: (%.2f, %.2f)", self.rrt_target.x, self.rrt_target.y)
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
            
            
            
        
        # 수신 못 했다면 기존처럼 멀리 있는 콘들을 목표점으로 함    
        else:
            # self.rrt_target이 없는 경우, frontCones 리스트에서 조건에 맞는 콘을 선택
            for cone in frontCones:
                coneDist = self.dist(self.carPosX, self.carPosY, cone.x, cone.y)
                if coneDist > 6:
                    rrtTarget.append((cone.x, cone.y, coneObstacleSize))
                    break  # 조건에 맞는 콘을 하나 찾으면 반복문 종료

            
        """트리 파라미터 조정 구간"""                

        start = [self.carPosX, self.carPosY, self.carPosYaw]
        iterationNumber = 100
        
        # RRT 경로 계획에서 최대 트리 가지 길이
        planDistance = 5.4
        
        # RRT 노드 간 이동 거리 (스텝 길이)
        expandDistance = 0.6
        
        # 다음 노드 생성 시 각도 제한 (회전 제한)
        expandAngle = 20


        """트리 파라미터 조정 구간""" 

        rrt = ma_rrt.RRT(start, planDistance, obstacleList=obstacleList, expandDis=expandDistance, turnAngle=expandAngle, maxIter=iterationNumber, rrtTargets = rrtTarget)
        nodeList, leafNodes = rrt.Planning()

        self.publishTreeVisual(nodeList, leafNodes)

        # 기본 전방 범위보다 약간 넓은 범위에서 콘들을 모아 경로 평가나 보완에 활용
        frontConesBiggerDist = 20
        largerGroupFrontCones = self.getFrontConeObstacles(self.map, frontConesBiggerDist)

        bestBranch = self.findBestBranch(leafNodes, nodeList, largerGroupFrontCones, coneObstacleSize, expandDistance, planDistance)

        if bestBranch:
            filteredBestBranch = self.getFilteredBestBranch(bestBranch)

            if filteredBestBranch:
                delaunayEdges = self.getDelaunayEdges(frontCones)
                self.publishDelaunayEdgesVisual(delaunayEdges)
                newWaypoints = []

                if delaunayEdges:
                    newWaypoints = self.getWaypointsFromEdges(filteredBestBranch, delaunayEdges)

                if newWaypoints:
                    self.mergeWaypoints(newWaypoints)
                
                
                # 새 웨이포인트가 전혀 없으면 False 발행
                else:
                    no_wp = Bool()
                    no_wp.data = False
                    self.obstacleExistencePub.publish(no_wp)
                            
                self.publishWaypoints(newWaypoints)



    def mergeWaypoints(self, newWaypoints):
        
        # 새 웨이포인트가 전혀 없으면 False 발행
        if not newWaypoints:
            no_wp = Bool()
            no_wp.data = False
            self.obstacleExistencePub.publish(no_wp)
            return

        # 차량의 현재 위치와 후보 웨이포인트 간의 거리가 2.0미터 이하일 때만 해당 웨이포인트를 저장 대상으로 고려
        maxDistToSaveWaypoints = 2.0
        
        # 새로운 웨이포인트 리스트에서 최대 2개까지만 저장 
        maxWaypointAmountToSave = 2
        
        # 기존에 저장된 웨이포인트와 새 후보 웨이포인트 간의 거리를 비교 -> 만약 두 점 간의 거리가 이 값보다 작으면, 두 점이 “거의 동일하다”고 판단하여 중복을 제거
        waypointsDistTollerance = 1000

        if len(self.savedWaypoints) > 15:
            firstSavedWaypoint = self.savedWaypoints[0]

            for waypoint in reversed(newWaypoints):
                distDiff = self.dist(firstSavedWaypoint[0], firstSavedWaypoint[1], waypoint[0], waypoint[1])
                if distDiff < waypointsDistTollerance:
                    self.preliminaryLoopClosure = False
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
                        self.savedWaypoints.remove(savedWaypoint)
                        break

                if (self.preliminaryLoopClosure):
                    distDiff = self.dist(firstSavedWaypoint[0], firstSavedWaypoint[1], waypointCandidate[0], waypointCandidate[1])
                    if distDiff < waypointsDistTollerance:
                        self.loopClosure = False
                        break

                self.savedWaypoints.append(waypointCandidate)
                newSavedPoints.append(waypointCandidate)

        if newSavedPoints:
            for point in newSavedPoints:
                newWaypoints.remove(point)

    def getWaypointsFromEdges(self, filteredBranch, delaunayEdges):
        if not delaunayEdges:
            return

        waypoints = []
        for i in range (len(filteredBranch) - 1):
            node1 = filteredBranch[i]
            node2 = filteredBranch[i+1]
            a1 = np.array([node1.x, node1.y])
            a2 = np.array([node2.x, node2.y])

            maxAcceptedEdgeLength = 7
            maxEdgePartsRatio = 3

            intersectedEdges = []
            for edge in delaunayEdges:

                b1 = np.array([edge.x1, edge.y1])
                b2 = np.array([edge.x2, edge.y2])

                # 1) RRT 브랜치(a1~a2)와 Delaunay 에지(b1~b2)가 교차하는지 확인
                if self.getLineSegmentIntersection(a1, a2, b1, b2):
                    
                    # 2) 에지 길이 및 분할 비율 제한
                    if edge.length() < maxAcceptedEdgeLength:
                        edge.intersection = self.getLineIntersection(a1, a2, b1, b2)

                        edgePartsRatio = edge.getPartsLengthRatio()

                        if edgePartsRatio < maxEdgePartsRatio:
                            intersectedEdges.append(edge)

            # 교차 에지가 있으면, 그 에지의 중간점(혹은 교차점)을 웨이포인트로 추가
            if intersectedEdges:

                if len(intersectedEdges) == 1:
                    edge = intersectedEdges[0]

                    waypoints.append(edge.getMiddlePoint())
                    
                # 여러 에지가 교차하면, 거리 순으로 정렬 후 모두 추가
                else:
                    intersectedEdges.sort(key=lambda edge: self.dist(node1.x, node1.y, edge.intersection[0], edge.intersection[1], shouldSqrt = False))

                    for edge in intersectedEdges:
                        waypoints.append(edge.getMiddlePoint())

        return waypoints

    def getDelaunayEdges(self, frontCones):
        # frontCones: TrackCone 리스트
        # 내부에서 compressedWallObstacleList까지 합침
        obstacles = []
        # TrackCone 타입이면 .x/.y, 아니면 (x,y,_) 튜플 처리
        for cone in frontCones:
            obstacles.append((cone.x, cone.y))
        for w in self.compressedWallObstacleList:
            obstacles.append((w[0], w[1]))
        if len(obstacles) < 4:
            return

        pts = np.array(obstacles)
        tri = Delaunay(pts)
        delaunayEdges = []
        for simp in tri.simplices:
            for i in range(3):
                j = (i + 1) % 3
                x1, y1 = pts[simp[i]]
                x2, y2 = pts[simp[j]]
                edge = Edge(x1, y1, x2, y2)
                if edge not in delaunayEdges:
                    delaunayEdges.append(edge)
        return delaunayEdges


    def dist(self, x1, y1, x2, y2, shouldSqrt = True):
        distSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return math.sqrt(distSq) if shouldSqrt else distSq

    def publishWaypoints(self, newWaypoints = None):
        if (time.time() - self.lastPublishWaypointsTime) < self.waypointsPublishInterval:
            return

        waypointsArray = WaypointsArray()
        newwaypointsArray = WaypointsArray()
        waypointsArray.header.frame_id = self.world_frame
        waypointsArray.header.stamp = rospy.Time.now()
        newwaypointsArray.header.frame_id = self.world_frame
        newwaypointsArray.header.stamp = rospy.Time.now()

        for i in range(len(self.savedWaypoints)):
            waypoint = self.savedWaypoints[i]
            waypointId = len(waypointsArray.waypoints)
            w = Waypoint(waypoint[0], waypoint[1], waypointId)
            waypointsArray.waypoints.append(w)

        if newWaypoints is not None:
            for i in range(len(newWaypoints)):
                waypoint = newWaypoints[i]
                waypointId = len(waypointsArray.waypoints)
                w = Waypoint(waypoint[0], waypoint[1], waypointId)
                waypointsArray.waypoints.append(w)
                newwaypointsArray.waypoints.append(w)

        if self.shouldPublishWaypoints:
            self.waypointsPub.publish(waypointsArray)
            self.newwaypointsPub.publish(newwaypointsArray)
            self.lastPublishWaypointsTime = time.time()
            self.publishWaypointsVisuals(newWaypoints)


    def publishWaypointsVisuals(self, newWaypoints = None):

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
            p = Point(waypoint[0], waypoint[1], 0.0)
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
                p = Point(waypoint[0], waypoint[1], 0.0)
                newWaypointsMarker.points.append(p)

            markerArray.markers.append(newWaypointsMarker)

        self.waypointsVisualPub.publish(markerArray)

    def getLineIntersection(self, a1, a2, b1, b2):
        s = np.vstack([a1,a2,b1,b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def getLineSegmentIntersection(self, a1, a2, b1, b2):
        return self.ccw(a1,b1,b2) != self.ccw(a2,b1,b2) and self.ccw(a1,a2,b1) != self.ccw(a1,a2,b2)

    def ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def getFilteredBestBranch(self, bestBranch):
        if not bestBranch:
            return

        everyPointDistChangeLimit = 2.0
        newPointFilter = 0.2
        maxDiscardAmountForReset = 2

        if not self.filteredBestBranch:
            self.filteredBestBranch = list(bestBranch)
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
                        self.filteredBestBranch = list(bestBranch)
                    break

                changeRate += (everyPointDistChangeLimit - dist)

            if not shouldDiscard:

                for i in range(len(bestBranch)):
                    self.filteredBestBranch[i].x = self.filteredBestBranch[i].x * (1 - newPointFilter) + newPointFilter * bestBranch[i].x
                    self.filteredBestBranch[i].y = self.filteredBestBranch[i].y * (1 - newPointFilter) + newPointFilter * bestBranch[i].y

                self.discardAmount = 0

        self.publishFilteredBranchVisual()
        return list(self.filteredBestBranch)

    def publishDelaunayEdgesVisual(self, edges):
        if not edges:
            return

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(1)
        marker.ns = "publishDelaunayLinesVisual"

        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.05

        marker.pose.orientation.w = 1

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.b = 1.0

        for edge in edges:
            # print edge

            p1 = Point(edge.x1, edge.y1, 0)
            p2 = Point(edge.x2, edge.y2, 0)

            marker.points.append(p1)
            marker.points.append(p2)

        self.delaunayLinesVisualPub.publish(marker)

    def findBestBranch(self, leafNodes, nodeList, largerGroupFrontCones, coneObstacleSize, expandDistance, planDistance):
        if not leafNodes:
            return

        coneDistLimit = 4.0
        coneDistanceLimitSq = coneDistLimit * coneDistLimit;

        bothSidesImproveFactor = 3
        minAcceptableBranchRating = 90

        leafRatings = []
        for leaf in leafNodes:
            branchRating = 0
            node = leaf

            while node.parent is not None:
                nodeRating = 0

                leftCones = []
                rightCones = []

                for cone in largerGroupFrontCones:
                    coneDistSq = ((cone.x - node.x) ** 2 + (cone.y - node.y) ** 2)

                    if coneDistSq < coneDistanceLimitSq:
                        actualDist = math.sqrt(coneDistSq)

                        if actualDist < coneObstacleSize:
                            continue

                        nodeRating += (coneDistLimit - actualDist)

                        if self.isLeftCone(node, nodeList[node.parent], cone):
                            leftCones.append(cone)
                        else:
                            rightCones.append(cone)

                if ((len(leftCones) == 0 and len(rightCones)) > 0 or (len(leftCones) > 0 and len(rightCones) == 0)):
                    nodeRating /= bothSidesImproveFactor

                if (len(leftCones) > 0 and len(rightCones) > 0):
                    nodeRating *= bothSidesImproveFactor

                nodeFactor = (node.cost - expandDistance)/(planDistance - expandDistance) + 1

                branchRating += nodeRating * nodeFactor
                node = nodeList[node.parent]

            leafRatings.append(branchRating)

        maxRating = max(leafRatings)
        maxRatingInd = leafRatings.index(maxRating)

        node = leafNodes[maxRatingInd]

        if maxRating < minAcceptableBranchRating:
            return

        self.publishBestBranchVisual(nodeList, node)

        reverseBranch = []
        reverseBranch.append(node)
        while node.parent is not None:
            node = nodeList[node.parent]
            reverseBranch.append(node)

        directBranch = []
        for n in reversed(reverseBranch):
            directBranch.append(n)

        return directBranch

    def isLeftCone(self, node, parentNode, cone):
        return ((node.x - parentNode.x) * (cone.y - parentNode.y) - (node.y - parentNode.y) * (cone.x - parentNode.x)) > 0;

    def publishBestBranchVisual(self, nodeList, leafNode):
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

        if not self.filteredBestBranch:
            return

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.2)
        marker.ns = "publisshFilteredBranchVisual"

        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.07

        marker.pose.orientation.w = 1

        marker.color.a = 1.0
        marker.color.b = 1.0

        for i in range(len(self.filteredBestBranch)):
            node = self.filteredBestBranch[i]
            p = Point(node.x, node.y, 0)
            if i != 0:
                marker.points.append(p)

            if i != len(self.filteredBestBranch) - 1:
                marker.points.append(p)

        self.filteredBranchVisualPub.publish(marker)

    def publishTreeVisual(self, nodeList, leafNodes):

        if not nodeList and not leafNodes:
            return

        markerArray = MarkerArray()

        # tree lines marker
        treeMarker = Marker()
        treeMarker.header.frame_id = self.world_frame
        treeMarker.header.stamp = rospy.Time.now()
        treeMarker.ns = "rrt"

        treeMarker.type = treeMarker.LINE_LIST
        treeMarker.action = treeMarker.ADD
        treeMarker.scale.x = 0.03

        treeMarker.pose.orientation.w = 1

        treeMarker.color.a = 1.0
        treeMarker.color.g = 0.7

        treeMarker.lifetime = rospy.Duration(0.2)

        for node in nodeList:
            if node.parent is not None:
                p = Point(node.x, node.y, 0)
                treeMarker.points.append(p)

                p = Point(nodeList[node.parent].x, nodeList[node.parent].y, 0)
                treeMarker.points.append(p)

        markerArray.markers.append(treeMarker)

        # leaves nodes marker
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

        # publis marker array
        self.treeVisualPub.publish(markerArray)

    def getFrontConeObstacles(self, map, frontDist):
        if not map:
            return []

        headingVector = self.getHeadingVector()
        headingVectorOrt = [-headingVector[1], headingVector[0]]

        behindDist = 1.0
        carPosBehindPoint = [self.carPosX - behindDist * headingVector[0], self.carPosY - behindDist * headingVector[1]]


        frontDistSq = frontDist ** 2

        frontConeList = []
        for cone in map:
            if (headingVectorOrt[0] * (cone.y - carPosBehindPoint[1]) - headingVectorOrt[1] * (cone.x - carPosBehindPoint[0])) < 0:
                if ((cone.x) ** 2 + (cone.y) ** 2) < frontDistSq:
                    frontConeList.append(cone)
        return frontConeList

    def getHeadingVector(self):
        headingVector = [1.0, 0]
        carRotMat = np.array([[math.cos(self.carPosYaw), -math.sin(self.carPosYaw)], [math.sin(self.carPosYaw), math.cos(self.carPosYaw)]])
        headingVector = np.dot(carRotMat, headingVector)
        return headingVector

    def getConesInRadius(self, map, x, y, radius):
        coneList = []
        radiusSq = radius * radius
        for cone in map:
            if ((cone.x - x) ** 2 + (cone.y - y) ** 2) < radiusSq:
                coneList.append(cone)
        return coneList
    
class Edge():
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.intersection = None

    def getMiddlePoint(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def length(self):
        return math.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)

    def getPartsLengthRatio(self):
        import math

        part1Length = math.sqrt((self.x1 - self.intersection[0]) ** 2 + (self.y1 - self.intersection[1]) ** 2)
        part2Length = math.sqrt((self.intersection[0] - self.x2) ** 2 + (self.intersection[1] - self.y2) ** 2)

        return max(part1Length, part2Length) / min(part1Length, part2Length)

    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2
             or self.x1 == other.x2 and self.y1 == other.y2 and self.x2 == other.x1 and self.y2 == other.y1)

    def __str__(self):
        return "(" + str(round(self.x1, 2)) + "," + str(round(self.y1,2)) + "),(" + str(round(self.x2, 2)) + "," + str(round(self.y2,2)) + ")"

    def __repr__(self):
        return str(self)



if __name__ == '__main__':

    maNode = MaRRTPathPlanNode()