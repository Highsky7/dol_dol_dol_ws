#!/usr/bin/env python3

import rospy
from MaRRTPathPlanNode import MaRRTPathPlanNode

def main():
    rospy.init_node('MaRRTPathPlanNode')
    maRRTPathPlanNode = MaRRTPathPlanNode()

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        maRRTPathPlanNode.sampleTree()
        rate.sleep()

if __name__ == '__main__':
    main()
