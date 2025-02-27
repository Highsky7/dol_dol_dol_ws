#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from datetime import datetime

from rtcm_msgs.msg import Message
from base64 import b64encode
from threading import Thread

# Python3에서 httplib 대신 http.client
from http.client import HTTPConnection, IncompleteRead
import http.client

# --- IncompleteRead 패치 (Python3) ---
def patch_http_response_read(func):
    def inner(*args):
        try:
            return func(*args)
        except IncompleteRead as e:
            return e.partial
    return inner

http.client.HTTPResponse.read = patch_http_response_read(http.client.HTTPResponse.read)
# --------------------------------------


class NtripConnect(Thread):
    def __init__(self, ntc):
        super(NtripConnect, self).__init__()
        self.ntc = ntc
        self.stop_flag = False

    def run(self):
        # HTTP 헤더 생성
        auth_string = f"{self.ntc.ntrip_user}:{self.ntc.ntrip_pass}"
        # Base64 인코딩 (bytes -> str)
        auth_b64 = b64encode(auth_string.encode('ascii')).decode('ascii')
        headers = {
            'Ntrip-Version': 'Ntrip/2.0',
            'User-Agent': 'NTRIP ntrip_ros',
            'Connection': 'close',
            'Authorization': 'Basic ' + auth_b64
        }

        # NMEA GGA 문장도 bytes로
        body_data = self.ntc.nmea_gga.encode('ascii')

        # HTTPConnection 생성
        connection = HTTPConnection(self.ntc.ntrip_server)
        connection.request('GET', '/' + self.ntc.ntrip_stream, body_data, headers)
        response = connection.getresponse()
        if response.status != 200:
            raise Exception(f"Connection failed with status {response.status}")

        buf = b""
        rmsg = Message()
        restart_count = 0

        while not self.stop_flag:
            data = response.read(1)
            if len(data) != 0:
                # RTCM 패킷 시작바이트 (0xD3 = 211) 확인
                if data[0] == 211:
                    buf += data
                    # length 필드(2바이트) 읽기
                    data = response.read(2)
                    buf += data
                    cnt = data[0] * 256 + data[1]

                    # 메시지 타입 판단용 2바이트 읽기
                    data = response.read(2)
                    buf += data
                    # RTCM 타입 = 12bit msg_type, 따라서 >> 4 또는 //16
                    typ = ((data[0] * 256) + data[1]) // 16
                    
                    # 콘솔에 로그 찍기 (추가)
                    print(str(datetime.now()), "length:", cnt, "type:", typ)

                    # 이어지는 cnt+1 바이트 읽기
                    cnt = cnt + 1
                    extra_data = response.read(cnt)
                    buf += extra_data

                    # ROS 메시지 채워서 퍼블리시
                    rmsg.message = buf
                    rmsg.header.seq += 1
                    rmsg.header.stamp = rospy.get_rostime()
                    self.ntc.pub.publish(rmsg)

                    # 버퍼 초기화
                    buf = b""
                else:
                    # 디버그용: RTCM 시작 바이트가 아닌 경우
                    # 원치 않으면 주석처리
                    # print("Non-RTCM byte:", data)
                    pass
            else:
                # 길이가 0인 데이터를 받으면, 재접속 시도
                restart_count += 1
                print("Zero length data. Restarting connection #", restart_count)
                connection.close()
                connection = HTTPConnection(self.ntc.ntrip_server)
                connection.request('GET', '/' + self.ntc.ntrip_stream, body_data, headers)
                response = connection.getresponse()
                if response.status != 200:
                    raise Exception(f"Connection failed with status {response.status}")
                buf = b""

        connection.close()


class NtripClient:
    def __init__(self):
        rospy.init_node('ntripclient', anonymous=True)

        self.rtcm_topic = rospy.get_param('~rtcm_topic', 'rtcm')

        self.ntrip_server = rospy.get_param('~ntrip_server')
        self.ntrip_user = rospy.get_param('~ntrip_user')
        self.ntrip_pass = rospy.get_param('~ntrip_pass')
        self.ntrip_stream = rospy.get_param('~ntrip_stream')
        self.nmea_gga = rospy.get_param('~nmea_gga')

        self.pub = rospy.Publisher(self.rtcm_topic, Message, queue_size=10)

        self.connection = NtripConnect(self)
        self.connection.start()

    def run(self):
        rospy.spin()
        if self.connection is not None:
            self.connection.stop_flag = True


if __name__ == '__main__':
    client = NtripClient()
    client.run()
