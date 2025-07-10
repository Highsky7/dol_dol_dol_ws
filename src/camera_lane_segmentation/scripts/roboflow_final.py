#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, argparse, cv2, torch, numpy as np
from math import atan2, sqrt
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs        # 실질 사용은 없지만 의존성 유지
from vehicle_msgs.msg import Track, TrackCone  # /track 메시지 처리를 위해 추가

# ────────── 유틸 함수 ────────── #
def polyfit_lane(ys, xs, order=2):
    if len(ys) < 5: return None
    try: return np.polyfit(ys, xs, order)
    except (np.linalg.LinAlgError, TypeError): return None

def morph_close(m, k=5):
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))

def remove_small(m, min_sz=300):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8)
    out=np.zeros_like(m)
    for i in range(1,n):
        if s[i,cv2.CC_STAT_AREA]>=min_sz: out[l==i]=255
    return out

def keep_top2(m,min_area=300):
    n,l,s,_=cv2.connectedComponentsWithStats(m,8)
    if n<=1: return np.zeros_like(m)
    comps=[(i,s[i,cv2.CC_STAT_AREA]) for i in range(1,n) if s[i,cv2.CC_STAT_AREA]>=min_area]
    comps.sort(key=lambda x:x[1],reverse=True)
    out=np.zeros_like(m)
    for i,_ in comps[:2]: out[l==i]=255
    return out

def final_filter(m): return keep_top2(remove_small(morph_close(m,5),10000),300)

def overlay_polyline(img,coeff,color=(0,0,255),step=4,th=2):
    if coeff is None: return img
    h,w=img.shape[:2]; pts=[]
    for y in range(0,h,step):
        x=np.polyval(coeff,y)
        if 0<=x<w: pts.append((int(x),int(y)))
    if len(pts)>1:
        cv2.polylines(img,[np.array(pts,np.int32)],False,color,th)
    return img

# ────────── 메인 클래스 ────────── #
class LaneFollowerNode:
    def __init__(self,opt):
        self.opt,opt.device=opt,opt.device.lower()
        self.bridge=CvBridge()
        self.device=torch.device(f"cuda:{opt.device}") if opt.device.isdigit() and torch.cuda.is_available() else torch.device("cpu")
        rospy.loginfo(f"[LaneFollower] device={self.device}")
        self.model=YOLO(opt.weights).to(self.device)

        p = np.load(opt.param_file)
        self.bev_h, self.bev_w = int(p['warp_h']), int(p['warp_w'])
        self.M=cv2.getPerspectiveTransform(p['src_points'],p['dst_points'])
        self.mpy,self.y_off,self.mpx=0.0038125,1.41,0.00240625   # px→m 변환 계수

        self.trk={'left':{'c':None,'age':0},'right':{'c':None,'age':0}}
        self.center={'c':None}; self.ALPHA,self.MAX_AGE=0.6,7
        
        # ==================== 수정된 부분 1: Ld 멤버 변수 추가 및 초기화 ==================== #
        self.L=0.73; self.T_MIN,self.T_MAX=0.4,0.6; self.L_MIN,self.L_MAX=1.75,2.35
        self.throttle=self.T_MIN
        # Ld(Lookahead distance)를 클래스 멤버로 관리하고, 안정적인 주행을 위해 최소값으로 초기화합니다.
        self.Ld = self.L_MIN
        # ==================== 수정된 부분 끝 =========================================== #

        # 교통 콘 위치 저장을 위한 변수
        self.cones = []  # 교통 콘의 (x, y) 좌표를 저장하는 리스트

        # 퍼블리셔와 서브스크라이버 설정
        self.pub_steer=rospy.Publisher("auto_steer_angle_lane",Float32,queue_size=1)  # 조향각 발행
        self.pub_status=rospy.Publisher("lane_detection_status",Bool,queue_size=1)   # 차선 감지 상태 발행
        self.pub_markers=rospy.Publisher("lane_markers",MarkerArray,queue_size=1)    # 차선 마커 발행
        self.pub_path=rospy.Publisher("center_path",Path,queue_size=1)               # 중앙 경로 발행
        self.pub_look=rospy.Publisher("lookahead_point",Marker,queue_size=1)         # 전방 주시점 발행
        self.pub_forced_rrt=rospy.Publisher("/forced_rrt", Bool, queue_size=1)       # 강제 RRT 발행
        rospy.Subscriber("/usb_cam/image_raw",Image,self.img_cb,queue_size=1,buff_size=2**24)  # 카메라 이미지 구독
        rospy.Subscriber("auto_throttle",Float32,self.thr_cb,queue_size=1)           # 스로틀 값 구독
        rospy.Subscriber("/track", Track, self.track_cb, queue_size=1)               # 교통 콘 데이터 구독

        self.static_br=tf2_ros.StaticTransformBroadcaster(); self.pub_static_tf()
        rospy.loginfo("[LaneFollower] ready")

    # --------- TF 설정 --------- #
    def pub_static_tf(self):
        tx=rospy.get_param("~camera_offset_x",0.0)
        ty=rospy.get_param("~camera_offset_y",0.0)
        tz=rospy.get_param("~camera_offset_z",0.0)
        t=TransformStamped(); t.header.stamp=rospy.Time.now()
        t.header.frame_id,t.child_frame_id="velodyne","camera"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z=tx,ty,tz
        t.transform.rotation.w=1.0
        self.static_br.sendTransform(t)
        rospy.loginfo(f"[TF] velodyne→camera ({tx:.2f},{ty:.2f},{tz:.2f})")

    # --------- 콜백 함수 --------- #
    # ==================== 수정된 부분 2: throttle 콜백 로직 변경 ==================== #
    def thr_cb(self, m):
        """
        스로틀 값과 Lookahead Distance를 업데이트합니다.
        수신된 스로틀 값이 T_MIN보다 낮을 경우 Ld 값을 업데이트하지 않고
        이전 값을 유지하여 주행 중 오실레이션을 방지합니다.
        """
        subscribed_throttle = m.data
        if subscribed_throttle >= self.T_MIN:
            # 유효한 스로틀 값이 들어오면, throttle과 Ld를 모두 업데이트합니다.
            self.throttle = np.clip(subscribed_throttle, self.T_MIN, self.T_MAX)
            self.Ld = self.L_MIN + (self.L_MAX - self.L_MIN) * ((self.throttle - self.T_MIN) / (self.T_MAX - self.T_MIN))
        # else: subscribed_throttle < self.T_MIN 이면 아무것도 하지 않음으로써
        # self.Ld가 이전 값을 유지하도록 합니다.
    # ==================== 수정된 부분 끝 =========================================== #

    def img_cb(self,msg):
        """ 이미지 수신 및 처리 """
        try: img=self.bridge.imgmsg_to_cv2(msg,"bgr8")
        except CvBridgeError as e: rospy.logerr(e); return
        self.process(img)
    
    def track_cb(self, msg):
        """ /track 토픽의 콜백 함수: 교통 콘의 위치를 저장 """
        self.cones = []  # 기존 콘 정보 초기화
        for cone in msg.cones:
            if cone.type == "traffic_cone":
                self.cones.append((cone.x, cone.y))  # (x, y) 좌표 저장

    # --------- 좌표 변환 --------- #
    def img2veh(self,u,v):
        """ 이미지 좌표를 차량 좌표로 변환 """
        x=(self.bev_h-v)*self.mpy+self.y_off
        y=(self.bev_w/2-u)*self.mpx
        return x,y

    # --------- 메인 처리 함수 --------- #
    def process(self,im):
        """ 이미지 처리 및 경로 생성 """
        bev=cv2.warpPerspective(im,self.M,(self.bev_w,self.bev_h))
        r=self.model(bev,imgsz=self.opt.img_size,conf=self.opt.conf_thres,
                     iou=self.opt.iou_thres,device=self.device,verbose=False)[0]

        mask=np.zeros(r.orig_shape,np.uint8)
        if r.masks is not None:
            for conf,m in zip(r.boxes.conf,r.masks.data):
                if conf<0.5: continue
                m=cv2.resize((m.cpu().numpy()*255).astype(np.uint8),(mask.shape[1],mask.shape[0]))
                mask=np.maximum(mask,m)
        mask=final_filter(mask)

        n,lbl,st,_=cv2.connectedComponentsWithStats(mask,8)
        dets=[]
        for i in range(1,n):
            if st[i,cv2.CC_STAT_AREA]<100: continue
            ys,xs=np.where(lbl==i)
            c=polyfit_lane(ys,xs);
            if c is None: continue
            dets.append({'c':c,'xb':np.polyval(c,self.bev_h-1)})
        dets.sort(key=lambda d:d['xb'])

        left,right=self.trk['left'],self.trk['right']; cur_l,cur_r=None,None
        if len(dets)==2: cur_l,cur_r=dets
        elif len(dets)==1:
            d=dets[0]
            dl=abs(d['xb']-np.polyval(left['c'],self.bev_h-1)) if left['c'] is not None else 1e9
            dr=abs(d['xb']-np.polyval(right['c'],self.bev_h-1)) if right['c'] is not None else 1e9
            if dl<dr: cur_l=d
            elif dr<dl: cur_r=d
            else: (cur_l if d['xb']<self.bev_w/2 else cur_r)==d  # noqa

        def update(trk,det):
            if det:
                trk['c']=det['c'] if trk['c'] is None else self.ALPHA*det['c']+(1-self.ALPHA)*trk['c']
                trk['age']=0
            else:
                trk['age']+=1
                if trk['age']>self.MAX_AGE: trk['c']=None
        update(left,cur_l); update(right,cur_r)
        cL,cR=left['c'],right['c']
        lane_ok=cL is not None or cR is not None
        self.pub_status.publish(Bool(data=lane_ok))

        # 중앙선 계산
        center_coeff=None
        if lane_ok:
            w_px=1.5/self.mpx
            pts=[]
            for y in range(self.bev_h-1,self.bev_h//2,-1):
                if cL is not None and cR is not None:
                    xc=(np.polyval(cL,y)+np.polyval(cR,y))/2
                elif cL is not None:
                    xc=np.polyval(cL,y)+w_px/2
                elif cR is not None:
                    xc=np.polyval(cR,y)-w_px/2
                else: continue
                pts.append((xc,y))
            if len(pts)>10:
                p=np.array(pts); center_coeff=polyfit_lane(p[:,1],p[:,0])
        if center_coeff is not None:
            self.center['c']=center_coeff if self.center['c'] is None else self.ALPHA*center_coeff+(1-self.ALPHA)*self.center['c']
        cC=self.center['c']

        # ==================== 수정된 부분 3: Ld 계산 로직 변경 ==================== #
        # thr_cb 콜백에서 관리되는 self.Ld 값을 직접 사용합니다.
        Ld = self.Ld
        # ==================== 수정된 부분 끝 =========================================== #
        
        goal=None; steer_deg=None
        if cC is not None:
            for y in range(self.bev_h-1,-1,-1):
                x=np.polyval(cC,y); xv,yv=self.img2veh(x,y)
                if sqrt(xv**2+yv**2)>=Ld:
                    goal=(xv,yv); break
            if goal:
                xv,yv=goal
                steer_deg=-np.degrees(atan2(2*self.L*yv,xv**2+yv**2))
                steer_deg=np.clip(steer_deg,-25,25)
                self.pub_steer.publish(Float32(data=steer_deg))

        # BEV 영역 내 콘 확인 및 /forced_rrt 발행
        x_min, x_max = 1.41, 2.95  # BEV x 경계 (미터 단위)
        y_min, y_max = -1.22, 1.22  # BEV y 경계 (미터 단위)
        cone_in_bev = any(x_min <= x <= x_max and y_min <= y <= y_max for x, y in self.cones)
        forced_rrt = lane_ok and cone_in_bev  # 차선과 콘 모두 있을 때 True
        self.pub_forced_rrt.publish(Bool(data=forced_rrt))

        # RViz에 시각화 데이터 발행
        self.publish_viz(cL,cR,cC,goal)

        # 디버그 정보 표시
        ann=r.plot()
        overlay_polyline(bev,cL,(0,0,255),2,2)  # 왼쪽 차선 빨간색
        overlay_polyline(bev,cR,(255,0,0),2,2)  # 오른쪽 차선 파란색
        if cC is not None: overlay_polyline(bev,cC,(0,255,0),2,3)  # 중앙선 초록색
        if goal: cv2.circle(bev,(int(self.bev_w/2 - goal[1]/self.mpx),int(self.bev_h - (goal[0]-self.y_off)/self.mpy)),10,(0,255,255),-1)
        txt=lambda s,y:cv2.putText(bev,s,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        txt(f"Steer: {steer_deg:.1f}" if steer_deg is not None else "Steer: N/A",30)  # 조향각
        txt(f"Lane Detected: {lane_ok}",60)  # 차선 감지 여부
        txt(f"Lookahead: {Ld:.2f}m",90)  # 전방 주시 거리
        txt(f"Throttle: {self.throttle:.2f}",120)  # 스로틀 값
        txt(f"Cones in BEV: {cone_in_bev}",150)  # BEV 내 콘 존재 여부
        txt(f"Forced RRT: {forced_rrt}",180)  # 강제 RRT 상태
        cv2.imshow("Original",im); cv2.imshow("Detections BEV",ann); cv2.imshow("Final Path BEV",bev); cv2.waitKey(1)

    # --------- RViz 시각화 함수 --------- #
    def publish_viz(self,cL,cR,cC,goal):
        """ RViz에 차선, 경로, BEV 영역 시각화 """
        now=rospy.Time.now()
        marr=MarkerArray()

        def lane_m(coeff,idx,color):
            m=Marker()
            m.header.stamp, m.header.frame_id=now,"camera"
            m.ns,m.id="lane",idx
            if coeff is None:
                m.action=Marker.DELETE; return m  #<-- 차선 없으면 삭제 명령
            m.action=Marker.ADD; m.type=Marker.LINE_STRIP; m.scale.x=0.05
            m.color.r,m.color.g,m.color.b,m.color.a=*color,1.0; m.pose.orientation.w=1.0
            for y in range(self.bev_h-1,self.bev_h//2,-8):
                x=np.polyval(coeff,y); xv,yv=self.img2veh(x,y)
                m.points.append(Point(x=xv,y=yv,z=0.0))
            return m
        marr.markers=[lane_m(cL,0,(1,0,0)), lane_m(cR,1,(0,0,1))]

        # BEV 영역 시각화
        bev_area_marker = Marker()
        bev_area_marker.header.stamp = now
        bev_area_marker.header.frame_id = "camera"
        bev_area_marker.ns = "bev_roi_area"
        bev_area_marker.id = 100
        bev_area_marker.type = Marker.LINE_STRIP
        bev_area_marker.action = Marker.ADD
        bev_area_marker.pose.orientation.w = 1.0
        bev_area_marker.scale.x = 0.03
        bev_area_marker.color.r, bev_area_marker.color.g, bev_area_marker.color.b, bev_area_marker.color.a = 0.0, 1.0, 1.0, 1.0
        bev_points = [
            Point(x=2.95, y=-1.22, z=0.0),  # 앞-오른쪽
            Point(x=2.95, y=1.22, z=0.0),   # 앞-왼쪽
            Point(x=1.41, y=1.22, z=0.0),   # 뒤-왼쪽
            Point(x=1.41, y=-1.22, z=0.0),  # 뒤-오른쪽
            Point(x=2.95, y=-1.22, z=0.0)   # 시작점으로 돌아가 닫힌 도형 형성
        ]
        bev_area_marker.points = bev_points
        marr.markers.append(bev_area_marker)

        self.pub_markers.publish(marr)

        # 중앙 경로 발행
        p=Path(); p.header.stamp,p.header.frame_id=now,"camera"
        if cC is not None:
            for y in range(self.bev_h-1,self.bev_h//2,-6):
                x=np.polyval(cC,y); xv,yv=self.img2veh(x,y)
                ps=PoseStamped(); ps.header=p.header; ps.pose.position.x,ps.pose.position.y=xv,yv; ps.pose.orientation.w=1.0
                p.poses.append(ps)
        self.pub_path.publish(p)

        # 전방 주시점 발행
        mk=Marker(); mk.header.stamp,mk.header.frame_id=now,"camera"; mk.ns="lookahead"; mk.id=0
        if goal is None:
            mk.action=Marker.DELETE
        else:
            mk.action=Marker.ADD; mk.type=Marker.SPHERE
            mk.scale.x=mk.scale.y=mk.scale.z=0.25
            mk.color.r,mk.color.g,mk.color.b,mk.color.a=1,1,0,1
            mk.pose.position.x,mk.pose.position.y=goal; mk.pose.orientation.w=1
        self.pub_look.publish(mk)

# ────────── 메인 함수 ────────── #
def main():
    """ 노드 초기화 및 실행 """
    rospy.init_node("lane_follower_node",anonymous=True)
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',type=str,default='./weights3.pt')
    ap.add_argument('--device',default='0')
    ap.add_argument('--img-size',type=int,default=640)
    ap.add_argument('--conf-thres',type=float,default=0.6)
    ap.add_argument('--iou-thres',type=float,default=0.5)
    ap.add_argument('--param-file',type=str,default='./bev_params_7.npz')
    opt,_=ap.parse_known_args()
    LaneFollowerNode(opt); rospy.spin()

if __name__=="__main__":
    main()