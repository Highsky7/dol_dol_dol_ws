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
    index, Long, Lat, UTM_X, UTM_Y, Cov_00 형식의 CSV 파일을 생성합니다.
    - Long: 경도, Lat: 위도
    - UTM_X, UTM_Y: utm.from_latlon()을 이용해 계산한 값
    - Cov_00: position_covariance[0] (x 방향 위치 분산)
    """
    if not os.path.exists(bag_filepath):
        rospy.logerr("Bag 파일이 존재하지 않습니다: %s", bag_filepath)
        sys.exit(1)

    bag = rosbag.Bag(bag_filepath, 'r')
    index = 0
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 헤더 추가 -----------------------------------------------★
        writer.writerow(["index", "Long", "Lat", "UTM_X", "UTM_Y", "Cov_00"])

        for topic, msg, t in bag.read_messages(topics=['/ublox_gps/fix']):
            try:
                lat = msg.latitude
                lon = msg.longitude
                # position_covariance 배열이 있고 길이가 9인지 확인 ----★
                if len(msg.position_covariance) == 9:
                    cov_00 = msg.position_covariance[0]
                else:
                    cov_00 = float('nan')  # 정보가 없으면 NaN 기록
            except AttributeError:
                rospy.logwarn("메시지에 위도/경도 또는 covariance 필드가 없습니다.")
                continue

            try:
                easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
            except Exception as e:
                rospy.logwarn("UTM 변환 오류: %s", str(e))
                continue

            # CSV 한 줄 작성 ---------------------------------------★
            writer.writerow([index, lon, lat, easting, northing, cov_00])
            index += 1

    bag.close()
    rospy.loginfo("CSV 파일 생성 완료: %s (총 %d 개의 데이터)", output_csv, index)

if __name__ == '__main__':
    rospy.init_node('course_csv_creator', anonymous=True)
    data_dir = "src/gps_to_utm_pkg/data"
    bag_filepath = os.path.join(data_dir, "nocheon.bag")
    output_csv = os.path.join(data_dir, "nocheon_with_cov.csv")

    if len(sys.argv) > 1:
        bag_filepath = sys.argv[1]

    rospy.loginfo("Bag 파일: %s", bag_filepath)
    rospy.loginfo("출력 CSV 파일: %s", output_csv)

    create_course_csv(bag_filepath, output_csv)
