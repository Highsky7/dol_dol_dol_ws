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
        self.shouldPublishWaypoints = rospy.get_param('~publishWaypoints', True)
        self.shouldPublishPredefined = rospy.get_param('~publishPredefined', False)

        # 파일 경로 설정
        if rospy.has_param('~path'):
            self.path = rospy.get_param('~path')
        if rospy.has_param('~filename'):
            self.filename = rospy.get_param('~filename')

        # 오도메트리 토픽 이름 설정
        self.odometry_topic = rospy.get_param('~odom_topic', "/odometry")

        # 월드 프레임 설정
        self.world_frame = rospy.get_param('~world_frame', "velodyne")

        # 웨이포인트 발행 주기 설정
        waypointsFrequency = rospy.get_param('~desiredWaypointsFrequency', 10)
        self.waypointsPublishInterval = 1.0 / waypointsFrequency
        self.lastPublishWaypointsTime = 0

        # 파라미터화된 변수들
        self.frontConesDist = rospy.get_param('~frontConesDist', 12.0)  # 트리 생성에 고려할 콘의 최대 거리 (미터)
        self.coneObstacleSize = rospy.get_param('~coneObstacleSize', 0.7)  # 콘 장애물 반지름 (미터)
        self.iterationNumber = rospy.get_param('~iterationNumber', 100)  # RRT 최대 반복 횟수
        self.planDistance = rospy.get_param('~planDistance', 5.4)  # 최대 트리 확장 거리 (미터)
        self.expandDistance = rospy.get_param('~expandDistance', 0.6)  # 노드 간 이동 거리 (스텝 크기)
        self.expandAngle = rospy.get_param('~expandAngle', 20)  # 노드 생성 각도 (도)
        self.coneDistLimit = rospy.get_param('~coneDistLimit', 4.0)  # 콘 거리 제한
        self.wallSafetyMargin = rospy.get_param('~wallSafetyMargin', 0.1)  # 벽과의 안전 거리 (미터)
        self.penalty_factor = rospy.get_param('~penalty_factor', 5.0)  # 벽 페널티 계수
        self.maxDistToSaveWaypoints = rospy.get_param('~maxDistToSaveWaypoints', 2.0)  # 차량과 웨이포인트 간 최대 거리
        self.maxWaypointAmountToSave = rospy.get_param('~maxWaypointAmountToSave', 4)  # 저장할 최대 웨이포인트 수
        self.newPointFilter = rospy.get_param('~newPointFilter', 0.2)  # 새 위치 반영 비율
        self.wallObstacleRadius = rospy.get_param('~wallObstacleRadius', 0.01)  # 벽 장애물 반지름 (미터)

        # 장애물 존재 여부 퍼블리셔
        self.obstacleExistencePub = rospy.Publisher("/obstacle_existence", Bool, queue_size=1)
        
        # 구독자 설정
        rospy.Subscriber("/track", Track, self.mapCallback)
        rospy.Subscriber(self.odometry_topic, Odometry, self.odometryCallback)
        rospy.Subscriber("/rrt_target", PointStamped, self.rrtTargetCallback)
        self.rrt_target = None
        rospy.Subscriber("/predicted_trajectory_endpoint", MarkerArray, self.predictedTrajectoryEndpointCallback)
        rospy.Subscriber("/compressed_wall", PointCloud2, self.compressedWallCallback)

        # 퍼블리셔 설정
        self.rrtTargetVisualPub = rospy.Publisher("/visual/rrt_target", Marker, queue_size=1)
        self.waypointsPub = rospy.Publisher("/waypoints", WaypointsArray, queue_size=0)
        self.newwaypointsPub = rospy.Publisher("/newwaypoints", WaypointsArray, queue_size=5)
        self.treeVisualPub = rospy.Publisher("/visual/tree_marker_array", MarkerArray, queue_size=0)
        self.bestBranchVisualPub = rospy.Publisher("/visual/best_tree_branch", Marker, queue_size=1)
        self.filteredBranchVisualPub = rospy.Publisher("/visual/filtered_tree_branch", Marker, queue_size=1)
        self.delaunayLinesVisualPub = rospy.Publisher("/visual/delaunay_lines", Marker, queue_size=1)
        self.waypointsVisualPub = rospy.Publisher("/visual/waypoints", MarkerArray, queue_size=1)
        self.obstacleVisualPub = rospy.Publisher("/visual/obstacle_radius", MarkerArray, queue_size=1)

        # 장애물 리스트 초기화
        self.predictedEndpointObstacleList = []
        self.compressedWallObstacleList = []
        
        # 차량 상태 초기화
        self.carPosX = 0.0
        self.carPosY = 0.0
        self.carPosYaw = 0.0

        # 기타 상태 변수 초기화
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
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = 0.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(1.0)
            markerArray.markers.append(marker)
        self.obstacleVisualPub.publish(markerArray)

    def rrtTargetCallback(self, msg):
        self.rrt_target = msg.point

    def __del__(self):
        print('MaRRTPathPlanNode: Destructor called.')

    def odometryCallback(self, odometry):
        from tf.transformations import euler_from_quaternion
        self.carPosX = 0.0
        self.carPosY = 0.0
        q = odometry.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.carPosYaw = yaw

    def mapCallback(self, track):
        self.map = track.cones

    def predictedTrajectoryEndpointCallback(self, msg):
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            self.predictedEndpointObstacleList.append((x, y, 0.4))

    def compressedWallCallback(self, msg):
        self.compressedWallObstacleList = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p[:3]
            self.compressedWallObstacleList.append((x, y, self.wallObstacleRadius))  # 파라미터화된 반지름 사용

    def sampleTree(self):
        self.coneObstacleList = []
        self.coneObstacleList.clear()
        
        if self.loopClosure and len(self.savedWaypoints) > 0:
            self.publishWaypoints()
            return

        frontCones = self.getFrontConeObstacles(self.map, self.frontConesDist)
        self.coneObstacleList = [(cone.x, cone.y, self.coneObstacleSize) for cone in frontCones]

        obstacleList = self.coneObstacleList + self.predictedEndpointObstacleList + self.compressedWallObstacleList
        
        exists = Bool()
        exists.data = (len(obstacleList) > 0)
        self.obstacleExistencePub.publish(exists)
        
        rospy.loginfo("-----")
        rospy.loginfo("coneObstacleList: %d", len(self.coneObstacleList))
        rospy.loginfo("predictedEndpointObstacleList: %d", len(self.predictedEndpointObstacleList))
        rospy.loginfo("compressedWallObstacleList: %d", len(self.compressedWallObstacleList))
        rospy.loginfo("Total obstacles: %d", len(obstacleList))
        if not self.map:
            rospy.logwarn("No track data received")
        if not self.compressedWallObstacleList:
            rospy.logwarn("No compressed wall data received")
        rospy.loginfo("-----")
        
        self.predictedEndpointObstacleList.clear()
        self.publishObstacleVisuals(obstacleList)

        rrtTarget = []
        targetRadius = 0.01
                
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
        else:
            for cone in frontCones:
                coneDist = self.dist(self.carPosX, self.carPosY, cone.x, cone.y)
                if coneDist > 6:
                    rrtTarget.append((cone.x, cone.y, self.coneObstacleSize))
                    break

        start = [self.carPosX, self.carPosY, self.carPosYaw]
        rrt = ma_rrt.RRT(start, self.planDistance, obstacleList=obstacleList, expandDis=self.expandDistance, turnAngle=self.expandAngle, maxIter=self.iterationNumber, rrtTargets=rrtTarget)
        nodeList, leafNodes = rrt.Planning()

        self.publishTreeVisual(nodeList, leafNodes)

        bestBranch = self.findBestBranch(leafNodes, nodeList, obstacleList, self.coneObstacleSize, self.expandDistance, self.planDistance)

        coneDelaunayEdges = self.getDelaunayEdges(self.coneObstacleList, "cone")
        wallDelaunayEdges = self.getDelaunayEdges(self.compressedWallObstacleList, "wall")
        delaunayEdges = coneDelaunayEdges + wallDelaunayEdges
        self.publishDelaunayEdgesVisual(delaunayEdges)
        
        newWaypoints = []
        if coneDelaunayEdges and bestBranch:
            filteredBestBranch = self.getFilteredBestBranch(bestBranch)
            if filteredBestBranch:
                newWaypoints = self.getWaypointsFromEdges(filteredBestBranch, coneDelaunayEdges)
        elif coneDelaunayEdges:
            rospy.loginfo("No bestBranch, skipping waypoint generation")

        if newWaypoints:
            self.mergeWaypoints(newWaypoints)
        else:
            no_wp = Bool()
            no_wp.data = False
            self.obstacleExistencePub.publish(no_wp)

        self.publishWaypoints(newWaypoints)

    def getDelaunayEdges(self, obstacleList, obstacle_type=""):
        obstacles = [(obs[0], obs[1]) for obs in obstacleList]
        rospy.loginfo("getDelaunayEdges (%s): total obstacles=%d", obstacle_type, len(obstacles))
        if len(obstacles) < 3:
            rospy.loginfo("getDelaunayEdges (%s): 장애물 개수 부족 (< 3), 삼각형 생성 생략", obstacle_type)
            return []
        pts = np.array(obstacles)
        try:
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
            rospy.loginfo("getDelaunayEdges (%s): 생성된 에지 수=%d", obstacle_type, len(delaunayEdges))
            return delaunayEdges
        except Exception as e:
            rospy.logwarn("델로네 삼각형 생성 실패 (%s): %s", obstacle_type, str(e))
            return []

    def findBestBranch(self, leafNodes, nodeList, obstacleList, coneObstacleSize, expandDistance, planDistance):
        if not leafNodes:
            return
        coneDistanceLimitSq = self.coneDistLimit * self.coneDistLimit
        epsilon = 0.01
        bothSidesImproveFactor = 3
        minAcceptableBranchRating = 90
        leafRatings = []
        for leaf in leafNodes:
            branchRating = 0
            node = leaf
            while node.parent is not None:
                nodeRating = 0
                leftObstacles = []
                rightObstacles = []
                for obs in obstacleList:
                    x, y, radius = obs
                    distSq = ((x - node.x) ** 2 + (y - node.y) ** 2)
                    if distSq < coneDistanceLimitSq:
                        actualDist = math.sqrt(distSq)
                        if actualDist < radius:
                            continue
                        nodeRating += (self.coneDistLimit - actualDist)
                        temp_cone = type('Cone', (), {'x': x, 'y': y})()
                        if self.isLeftCone(node, nodeList[node.parent], temp_cone):
                            leftObstacles.append(obs)
                        else:
                            rightObstacles.append(obs)
                for obs in obstacleList:
                    x, y, radius = obs
                    wallDistSq = (x - node.x) ** 2 + (y - node.y) ** 2
                    if wallDistSq < (radius + self.wallSafetyMargin) ** 2:
                        nodeRating -= self.penalty_factor / (wallDistSq + epsilon)
                if (len(leftObstacles) == 0 and len(rightObstacles) > 0) or (len(leftObstacles) > 0 and len(rightObstacles) == 0):
                    nodeRating /= bothSidesImproveFactor
                if len(leftObstacles) > 0 and len(rightObstacles) > 0:
                    nodeRating *= bothSidesImproveFactor
                nodeFactor = (node.cost - expandDistance) / (planDistance - expandDistance) + 1
                branchRating += nodeRating * nodeFactor
                node = nodeList[node.parent]
            leafRatings.append(branchRating)
        maxRating = max(leafRatings)
        maxRatingInd = leafRatings.index(maxRating)
        node = leafNodes[maxRatingInd]
        if maxRating < minAcceptableBranchRating:
            return
        self.publishBestBranchVisual(nodeList, node)
        reverseBranch = [node]
        while node.parent is not None:
            node = nodeList[node.parent]
            reverseBranch.append(node)
        directBranch = [n for n in reversed(reverseBranch)]
        return directBranch

    def isLeftCone(self, node, parentNode, cone):
        return ((node.x - parentNode.x) * (cone.y - parentNode.y) - (node.y - parentNode.y) * (cone.x - parentNode.x)) > 0

    def dist(self, x1, y1, x2, y2, shouldSqrt=True):
        distSq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return math.sqrt(distSq) if shouldSqrt else distSq

    def mergeWaypoints(self, newWaypoints):
        if not newWaypoints:
            no_wp = Bool()
            no_wp.data = False
            self.obstacleExistencePub.publish(no_wp)
            return
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
            if i >= self.maxWaypointAmountToSave or carWaypointDist > self.maxDistToSaveWaypoints:
                break
            else:
                for savedWaypoint in reversed(self.savedWaypoints):
                    waypointsDistDiff = self.dist(savedWaypoint[0], savedWaypoint[1], waypointCandidate[0], waypointCandidate[1])
                    if waypointsDistDiff < waypointsDistTollerance:
                        self.savedWaypoints.remove(savedWaypoint)
                        break
                if self.preliminaryLoopClosure:
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
            return []
        waypoints = []
        for i in range(len(filteredBranch) - 1):
            node1 = filteredBranch[i]
            node2 = filteredBranch[i + 1]
            a1 = np.array([node1.x, node1.y])
            a2 = np.array([node2.x, node2.y])
            maxAcceptedEdgeLength = 7
            maxEdgePartsRatio = 3
            intersectedEdges = []
            for edge in delaunayEdges:
                b1 = np.array([edge.x1, edge.y1])
                b2 = np.array([edge.x2, edge.y2])
                if self.getLineSegmentIntersection(a1, a2, b1, b2):
                    if edge.length() < maxAcceptedEdgeLength:
                        edge.intersection = self.getLineIntersection(a1, a2, b1, b2)
                        edgePartsRatio = edge.getPartsLengthRatio()
                        if edgePartsRatio < maxEdgePartsRatio:
                            intersectedEdges.append(edge)
            if intersectedEdges:
                if len(intersectedEdges) == 1:
                    edge = intersectedEdges[0]
                    waypoints.append(edge.getMiddlePoint())
                else:
                    intersectedEdges.sort(key=lambda edge: self.dist(node1.x, node1.y, edge.intersection[0], edge.intersection[1], shouldSqrt=False))
                    for edge in intersectedEdges:
                        waypoints.append(edge.getMiddlePoint())
        return waypoints

    def publishWaypoints(self, newWaypoints=None):
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

    def publishWaypointsVisuals(self, newWaypoints=None):
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
        s = np.vstack([a1, a2, b1, b2])
        h = np.hstack((s, np.ones((4, 1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z == 0:
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def getLineSegmentIntersection(self, a1, a2, b1, b2):
        return self.ccw(a1, b1, b2) != self.ccw(a2, b1, b2) and self.ccw(a1, a2, b1) != self.ccw(a1, a2, b2)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def getFilteredBestBranch(self, bestBranch):
        if not bestBranch:
            return
        everyPointDistChangeLimit = 2.0
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
                    self.filteredBestBranch[i].x = self.filteredBestBranch[i].x * (1 - self.newPointFilter) + self.newPointFilter * bestBranch[i].x
                    self.filteredBestBranch[i].y = self.filteredBestBranch[i].y * (1 - self.newPointFilter) + self.newPointFilter * bestBranch[i].y
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
        marker.scale.x = 0.03
        marker.pose.orientation.w = 1
        marker.color.a = 0.3
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
        marker.lifetime = rospy.Duration(0.3)
        marker.ns = "publishFilteredBranchVisual"
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.pose.orientation.w = 1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0
        marker.color.b = 0
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
        treeMarker = Marker()
        treeMarker.header.frame_id = self.world_frame
        treeMarker.header.stamp = rospy.Time.now()
        treeMarker.ns = "rrt"
        treeMarker.type = treeMarker.LINE_LIST
        treeMarker.action = treeMarker.ADD
        treeMarker.scale.x = 0.03
        treeMarker.pose.orientation.w = 1
        treeMarker.color.a = 0.7
        treeMarker.color.r = 0.0
        treeMarker.color.g = 122.0 / 255.0
        treeMarker.color.b = 204.0 / 255.0
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
        leavesMarker.scale.x = 0.05
        leavesMarker.scale.y = 0.05
        leavesMarker.scale.z = 0.05
        leavesMarker.color.a = 0.7
        leavesMarker.color.r = 0.0
        leavesMarker.color.g = 122.0 / 255.0
        leavesMarker.color.b = 204.0 / 255.0
        for node in leafNodes:
            p = Point(node.x, node.y, 0)
            leavesMarker.points.append(p)
        markerArray.markers.append(leavesMarker)
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

class Edge:
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
        return f"({round(self.x1, 2)}, {round(self.y1, 2)}), ({round(self.x2, 2)}, {round(self.y2, 2)})"

    def __repr__(self):
        return str(self)

if __name__ == '__main__':
    maNode = MaRRTPathPlanNode()