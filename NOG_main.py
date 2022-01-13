from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from bvh import Bvh
from simple_viewer import SimpleViewer
from scipy.spatial.transform import Rotation as R

import math
import sys
import numpy as np
import dartpy as dart
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QOpenGLWidget
import platform

class MainWindow(QOpenGLWidget):

    def  __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.setMouseTracking(True)

        ## skel 파일 읽기 및 World 및 BodyNode, Joint 관련 정보 추출하기
        self.world=dart.utils.SkelParser.readWorld(os.path.abspath('assignment.skel'))
        self.Human=self.world.getSkeleton(0)
        self.Numbody=self.Human.getNumBodyNodes()
        self.Numjoint=self.Human.getNumJoints()
        self.Joint_name=list()
        for i in range(self.Numjoint):
            self.Joint_name.append(self.Human.getJoint(i).getName())
        
        ## 카메라 이동 관련 변수들
        self.L_mouse=False
        self.R_mouse=False
        self.H_mouse=False
        self.radius=7
        self.v_angle=math.pi/3
        self.h_angle=math.pi/7
        self.focus=np.array([0,-0.1,0])

        ## 시뮬레이션 Frame 및 Gain 관련 함수들
        self.timer=QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)
        self.frame=0
        
        ## 시뮬레이션 포즈 관련 변수들
        ## PD Contorl
        
        ## SPD Control
        self.BVH_pose = np.zeros(48)
        self.Num=0
        self.BVH_read()
 
    def initializeGL(self):
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)
 
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        ambient = [1.0,1.0,1.0,1.0]
        diffuse=[1.0,1.0,1.0,0.2]
        specular=[1.0,1.0,1.0,0.2]
        position=[5.0,5.0,5.0,5.0]
        
        mat_ambient=[0.5,0.5,0.5,0.0]
        mat_diffuse=[0.6,0.6,0.6,0.0]
        mat_specular=[0.7,0.7,0.7,0.0]
        mat_emissive=[0.0,0.0,0.0,0.0]
        mat_shininess=[30.0]
        
        glPushMatrix()
        glPushMatrix()
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular)
        glLightfv(GL_LIGHT0, GL_POSITION, position)
        glEnable(GL_LIGHT0)
        glPopMatrix()

        glPushMatrix()
        glEnable(GL_COLOR_MATERIAL)
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,mat_ambient)
        glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_diffuse)
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,mat_specular)
        glMaterialfv(GL_FRONT_AND_BACK,GL_SHININESS,mat_shininess)
        glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,mat_emissive)
        glPopMatrix()
        glPopMatrix()


        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
 
        glEnable(GL_TEXTURE_2D)
 
    def resizeGL(self, width, height):
        glGetError()
 
        aspect = width if (height == 0) else width / height
 
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def Camera(self):
        R_camera=np.array([self.radius*np.cos(self.v_angle)*np.cos(self.h_angle),self.radius*np.sin(self.h_angle),self.radius*np.sin(self.v_angle)*np.cos(self.h_angle)])
        R_plane=np.array([self.radius*np.sin(self.v_angle)*np.cos(self.h_angle),0,-self.radius*np.cos(self.v_angle)*np.cos(self.h_angle)])
        Upvec=np.cross(R_plane,-R_camera)
        
        glLoadIdentity()
        gluLookAt(R_camera[0]+self.focus[0],R_camera[1]+self.focus[1],R_camera[2]+self.focus[2],self.focus[0],self.focus[1],self.focus[2],Upvec[0],Upvec[1],Upvec[2])
    
    def Skel_Camera(self):
        R_camera = np.zeros(3)
        Normal = np.array([0,0,3])
        Vertical = np.array([1,0,0])
        
        focus = np.array(self.Human.getBodyNode(0).getWorldTransform().translation())
        angle = np.array(self.Human.getJoint(0).getPositions())
        
        Z_angle=angle[0]
        X_angle=angle[1]
        Y_angle=angle[2]
        
        Z_ROT=np.array([[np.cos(Z_angle),-np.sin(Z_angle),0],
                                [np.sin(Z_angle),np.cos(Z_angle),0],
                                [0,0,1]])
        X_ROT=np.array([[1,0,0],
                                [0,np.cos(X_angle),-np.sin(X_angle)],
                                [0,np.sin(X_angle),np.cos(X_angle)]])
        Y_ROT=np.array([[np.cos(Y_angle),0,np.sin(Y_angle)],
                                [0,1,0],
                                [-np.sin(Y_angle),0,np.cos(Y_angle)]])
        
        Normal = Z_ROT @ X_ROT @ Y_ROT @ Normal
        Vertical = Z_ROT @ X_ROT @ Y_ROT @ Vertical
        
        for i in range(3):
            R_camera[i] = focus[i] + Normal[i]
        Upvec=np.cross(Vertical,-Normal)
        
        glLoadIdentity()
        gluLookAt(R_camera[0],R_camera[1],R_camera[2],focus[0],focus[1],focus[2],0,1,0)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.Camera()
        #self.Skel_Camera()
        glPushMatrix()
        self.world.step()
        self.Drawskeleton()
        self.Drawbaseline()
        
        glLineWidth(2.0)
        glBegin(GL_LINES)    
        glColor3f(0,0,1)
        glVertex3fv([0,1,0])
        glVertex3fv([0,0,0])
        glEnd()
        glPopMatrix()
        glFlush()
        
        '''
        for i in range(90):
            self.world.step()
            self.PD_control()
        '''
        
        
        for i in range(120):
            self.SPD_control()
            self.world.step()
        
        
        if self.frame >= 198:
            self.frame =0
        elif self.frame < 198 :
            self.frame+=1
            
    def Drawbaseline(self):
        for i in range(200):
            glLineWidth(2.0)
            glBegin(GL_LINES)
            if i==100:
                glColor3f(1,1,0) # x축
                glVertex3fv([1,0,0])
                glVertex3fv([0,0,0])
                glColor3f(0,1,0) # z축
                glVertex3fv([0,0,1])
                glVertex3fv([0,0,0])
                
                glColor3f(1,1,1)
                glVertex3fv([100,0,0])
                glVertex3fv([1,0,0])
                glVertex3fv([0,0,0])
                glVertex3fv([-100,0,0])
                glVertex3fv([0,0,100])
                glVertex3fv([0,0,1])
                glVertex3fv([0,0,0])
                glVertex3fv([0,0,-100])
            else :
                glColor3f(1,1,1)
                glVertex3fv([-100+i,0,-100])
                glVertex3fv([-100+i,0,100])
                glVertex3fv([100,0,-100+i])
                glVertex3fv([-100,0,-100+i])
            glEnd()

    def Drawskeleton(self):
        for i in range(self.Numbody):
            Body=np.array(self.Human.getBodyNode(i).getWorldTransform().translation())
            Scale=np.array(self.Human.getBodyNode(i).getShapeNode(0).getShape().getSize())
            Rotation=np.array(self.Human.getBodyNode(i).getWorldTransform().rotation())
            Euler=self.cal_Rotation(Rotation)
            glPushMatrix()
            glTranslatef(Body[0],Body[1],Body[2])
            glMultMatrixf(Euler.T)
            glScalef(Scale[0],Scale[1],Scale[2])
            #self.DrawUnit()
            self.Drawunitbox()
            glPopMatrix()
            glFlush()

    def cal_Rotation(self,Rotation):
        Real_Rotation=np.zeros((4,4))
        for i in range(3):
            for j in range(3):
                Real_Rotation[i][j]=Rotation[i][j]
        Real_Rotation[3][3]=1
        return Real_Rotation
    
    def PD_control(self):
        Kp = 19
        Kd = 0.8
        
        for i in range(1,self.Numjoint):
            
            Z_angle=self.BVH_pose[self.frame][3*i+3]*math.pi/180
            X_angle=self.BVH_pose[self.frame][3*i+4]*math.pi/180
            Y_angle=self.BVH_pose[self.frame][3*i+5]*math.pi/180
            
            Z_ROT=np.array([[np.cos(Z_angle),-np.sin(Z_angle),0],
                                [np.sin(Z_angle),np.cos(Z_angle),0],
                                [0,0,1]])
            X_ROT=np.array([[1,0,0],
                                [0,np.cos(X_angle),-np.sin(X_angle)],
                                [0,np.sin(X_angle),np.cos(X_angle)]])
            Y_ROT=np.array([[np.cos(Y_angle),0,np.sin(Y_angle)],
                                [0,1,0],
                                [-np.sin(Y_angle),0,np.cos(Y_angle)]])
                
            ## ZXY 순으로 설정
            ## Rotation Matrix를 만들어야한다
            ## SKel 파일은 T Pose 형식으로 만들어서 진행을 시켜야 한다
            ## 각가의 Joint에 대해서 진행을 해보고 판단을 해보자
            position=np.array(self.Human.getJoint(i).getPositions())
            velocity=np.array(self.Human.getJoint(i).getVelocities())
            
            ROTR=(R.from_rotvec(position)).as_matrix()
            ROTD=Z_ROT@X_ROT@Y_ROT
            LOG=(R.from_matrix(ROTR.T@ROTD)).as_rotvec()
            torque=Kp*LOG+Kd*(0-velocity)
            self.Human.getJoint(i).setForces(np.array(torque))
                    
    def SPD_control(self):
        Kp=300
        Kd=0.4
        
        D_position=np.zeros(48)
        
        for i in range(48):
            D_position[i] = self.BVH_pose[self.frame][i]*math.pi/180
        
        self.Change_SPD(D_position)
        
        position = self.Human.getPositions()
        velocity = self.Human.getVelocities()
                    
        invM = np.linalg.inv(self.Human.getMassMatrix() + Kd * self.world.getTimeStep())
        PP = -Kp * (position + velocity * self.world.getTimeStep() - D_position)
        DD = -Kd * velocity
        QDDOT = invM @ (-self.Human.getCoriolisAndGravityForces() + PP + DD + self.Human.getConstraintForces())
        Torque = PP + DD - Kd * QDDOT * self.world.getTimeStep()
        
        for i in range(6):
            Torque[i] = 0
        
        self.Human.setForces(Torque)
            
    def BVH_read(self):
        f = open("sample-walk.bvh",'r')
        while self.Num < 99:
            lines = f.readline()
            self.Num +=1
        if self.Num >=99:
            while self.Num <298 :
                lines = f.readline()
                lines = lines.split()
                lines = np.array(list(map(float,lines))).T
                self.BVH_pose = np.vstack([self.BVH_pose,lines])
                self.Num +=1
            self.BVH_pose_PD = np.delete(self.BVH_pose,(0),axis=0)
        f.close()
        
    def Change_SPD(self,D_position):
        ## BVH 파일에서의 ZROT, XROT, YROT 순서를
        ## XROT, YROT, ZROT 순서로 바꾸어야 한다
        ## D_position은 (48,)
        ## (3 ~ 45까지 바꿔야 한다.)
        ## i= 1 ~ 15까지 진행한다.
        for i in range(1,16):
            Num = D_position[3*i] 
            D_position[3*i] = D_position[3*i+1]
            D_position[3*i+1] = D_position[3*i+2]
            D_position[3*i+2] = Num

    
    def DrawUnit(self):
        Unit=np.array([
            [1,0,0],
            [0.5, -0.5, 0.5],
            [1,0,0],
            [0.5, -0.5, -0.5],
            [1,0,0],
            [0.5,0.5,-0.5],
            
            [1,0,0],
            [0.5,0.5,-0.5],
            [1,0,0],
            [0.5,0.5,0.5],
            [1,0,0],
            [0.5,-0.5,0.5],
            
            [0,1,0],
            [0.5,0.5,0.5],
            [0,1,0],
            [0.5,0.5,-0.5],
            [0,1,0],
            [-0.5,0.5,0.5],
            
            [0,1,0],
            [-0.5,0.5,0.5],
            [0,1,0],
            [0.5,0.5,-0.5],
            [0,1,0],
            [-0.5,0.5,-0.5],
            
            [0,0,1],
            [0.5,0.5,0.5],
            [0,0,1],
            [-0.5,0.5,0.5],
            [0,0,1],
            [-0.5,-0.5,0.5],
            
            [0,0,1],
            [-0.5,-0.5,0.5],
            [0,0,1],
            [0.5,-0.5,0.5],
            [0,0,1],
            [0.5,0.5,0.5],
            
            [-1,0,0],
            [-0.5,0.5,0.5],
            [-1,0,0],
            [-0.5,0.5,-0.5],
            [-1,0,0],
            [-0.5,-0.5,-0.5],
            
            [-1,0,0],
            [-0.5,-0.5,-0.5],
            [-1,0,0],
            [-0.5,-0.5,0.5],
            [-1,0,0],
            [-0.5,0.5,0.5],
            
            [0,-1,0],
            [-0.5,-0.5,0.5],
            [0,-1,0],
            [0.5,-0.5,-0.5],
            [0,-1,0],
            [0.5,-0.5,0.5],
            
            [0,-1,0],
            [-0.5,-0.5,0.5],
            [0,-1,0],
            [-0.5,-0.5,-0.5],
            [0,-1,0],
            [0.5,-0.5,-0.5],
            
            [0,0,-1],
            [-0.5,0.5,-0.5],
            [0,0,-1],
            [0.5,0.5,-0.5],
            [0,0,-1],
            [-0.5,-0.5,-0.5],
            
            [0,0,-1],
            [0.5,0.5,-0.5],
            [0,0,-1],
            [0.5,-0.5,-0.5],
            [0,0,-1],
            [-0.5,-0.5,-0.5]
        ])
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT,6*Unit.itemsize,Unit)
        glVertexPointer(3, GL_FLOAT, 6*Unit.itemsize,ctypes.c_void_p(Unit.ctypes.data+3*Unit.itemsize))
        glDrawArrays(GL_TRIANGLES,0,int(Unit.size/6))
        
    
    # 기존에 있떤 연산속도가 느린 유닛 박스 그리기 방법
    def Drawunitbox(self):
        glBegin(GL_QUADS)
        glNormal3f(1,0,0)
        glColor3f(0.5,0.5,0.0)
        glVertex3f(0.5,-0.5,-0.5)
        glVertex3f(0.5,-0.5,0.5)
        glVertex3f(0.5,0.5,0.5)
        glVertex3f(0.5,0.5,-0.5)
        glEnd()

        glBegin(GL_QUADS)
        glNormal3f(0,0,-1)
        glVertex3f(0.5,0.5,-0.5)
        glVertex3f(-0.5,0.5,-0.5)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f(0.5,-0.5,-0.5)
        glEnd()

        glBegin(GL_QUADS)
        glNormal3f(-1,0,0)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f(-0.5,0.5,-0.5)
        glVertex3f(-0.5,0.5,0.5)
        glVertex3f(-0.5,-0.5,0.5)
        glEnd()

        glBegin(GL_QUADS)
        glNormal3f(0,0,1)
        glVertex3f(0.5,0.5,0.5)
        glVertex3f(0.5,-0.5,0.5)
        glVertex3f(-0.5,-0.5,0.5)
        glVertex3f(-0.5,0.5,0.5)
        glEnd()

        glBegin(GL_QUADS)
        glNormal3f(0,1,0)
        glVertex3f(0.5,0.5,0.5)
        glVertex3f(-0.5,0.5,0.5)
        glVertex3f(-0.5,0.5,-0.5)
        glVertex3f(0.5,0.5,-0.5)
        glEnd()

        glBegin(GL_QUADS)
        glNormal3f(0,-1,0)
        glVertex3f(0.5,-0.5,0.5)
        glVertex3f(0.5,-0.5,-0.5)
        glVertex3f(-0.5,-0.5,-0.5)
        glVertex3f(-0.5,-0.5,0.5)
        glEnd() 
        
    ''' 마우스 및 키보드 관련된 함수들 모음   
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.Human.getJoint('RightArm').setForces([100,0,0])
        elif event.key() == Qt.Key_Down:
            self.Human.getJoint('RightArm').setForces([-100,0,0])
        elif event.key() == Qt.Key_Left:
            self.Human.getJoint('RightArm').setForces([0,0,100])
        elif event.key() == Qt.Key_Right:
            self.Human.getJoint('RightArm').setForces([0,0,-100])
        super().keyPressEvent(event)
    '''
    '''    
    def mouseMoveEvent(self,event):
        if self.L_mouse == True or self.R_mouse == True:
            print('Mouse move {}: [{},{}]'.format(event.button(),event.x(),event.y()))

    def mousePressEvent(self,event):
        if event.button() == 1:
            self.L_mouse=True
        elif event.button() == 2:
            self.R_mouse=True

    def mouseReleaseEvent(self,event):
        if event.button() == 1:
            self.L_mouse=False
        elif event.button() == 2:
            self.R_mouse=False
    '''    
    def wheelEvent(self,event):
        self.radius+= event.angleDelta().y()/480
        
    def onStartButtonClicked(self):
        self.timer.start()
        self.btnStop.setEnabled(True)
        self.btnStart.setEnabled(False)
    def onStopButtonClicked(self):
        self.timer.stop()
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('MainWindow')
    window.setFixedSize(600,600)
    window.show()
    sys.exit(app.exec_())