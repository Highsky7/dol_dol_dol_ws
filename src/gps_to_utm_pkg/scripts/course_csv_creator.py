#!/usr/bin/env python3
import rosbag
import csv
import os
import sys
import rospy
import utm

def create_course_csv(bag_filepath, output_csv):
    """
    bag 파일에서 /ublox_gps/fix 토픽의 데이터를 읽어,
    index, Long, Lat, UTM_X, UTM_Y 형식의 CSV 파일을 생성합니다.
    - Long: 경도, Lat: 위도
    - UTM_X, UTM_Y: utm.from_latlon()을 이용해 계산한 값
    """
    # bag 파일이 존재하는지 확인
    if not os.path.exists(bag_filepath):
        rospy.logerr("Bag 파일이 존재하지 않습니다: %s", bag_filepath)
        sys.exit(1)
    
    bag = rosbag.Bag(bag_filepath, 'r')
    index = 0
    # CSV 파일 생성 (덮어쓰기)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 헤더 작성: index, Long, Lat, UTM_X, UTM_Y
        writer.writerow(["index", "Long", "Lat", "UTM_X", "UTM_Y"])
        
        # /ublox_gps/fix 토픽 메시지를 순회
        for topic, msg, t in bag.read_messages(topics=['/ublox_gps/fix']):
            # NavSatFix 메시지에서 위도와 경도 추출
            # 주의: 메시지 구조에 따라 latitude와 longitude 필드 사용
            try:
                lat = msg.latitude
                lon = msg.longitude
            except AttributeError:
                rospy.logwarn("메시지에 위도/경도 필드가 없습니다.")
                continue
            
            # UTM 변환 (위도, 경도 순으로)
            try:
                easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
            except Exception as e:
                rospy.logwarn("UTM 변환 오류: %s", str(e))
                continue
            
            # CSV에 기록: index, 경도, 위도, UTM_X, UTM_Y
            writer.writerow([index, lon, lat, easting, northing])
            index += 1

    bag.close()
    rospy.loginfo("CSV 파일 생성 완료: %s (총 %d 개의 데이터)", output_csv, index)

if __name__ == '__main__':
    rospy.init_node('course_csv_creator', anonymous=True)
    # bag 파일 경로와 출력 CSV 파일 경로를 지정합니다.
    # 기본적으로 git/gnss/src/gps_to_utm_pkg/data 폴더 내에 있다고 가정합니다.
    data_dir = "src/gps_to_utm_pkg/data"
    # 예를 들어, bag 파일 이름이 "contest_data.bag"라면:
    bag_filepath = os.path.join(data_dir, "smart_factory2.bag")
    # 출력 CSV 파일은 course1.csv로 저장
    output_csv = os.path.join(data_dir, "smart_factory2.csv")
    
    # 만약 명령행 인자로 bag 파일 경로를 전달하고 싶다면
    if len(sys.argv) > 1:
        bag_filepath = sys.argv[1]
    rospy.loginfo("Bag 파일: %s", bag_filepath)
    rospy.loginfo("출력 CSV 파일: %s", output_csv)
    
    create_course_csv(bag_filepath, output_csv)
