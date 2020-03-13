#!/usr/bin/env python3

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize

from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor

from random import random

import numpy as np
from numpy import sin, cos, pi

# A helper class that implements a slider with given start and end value; displays values
class SliderDisplay(QWidget):
    gui = None

    def __init__(self, name, low, high, initial_value, ticks=500):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        :param ticks: Resolution of slider - all sliders are integer/fixed number of ticks
        """
        # Save input values
        self.name = name
        self.low = low
        self.range = high - low
        self.ticks = ticks

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        # call back - call change_value when slider changed
        self.slider.valueChanged.connect(self.change_value)

        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return (self.slider.value() / self.ticks) * self.range + self.low

    # Called when the value changes - resets text
    def change_value(self):
        if (SliderDisplay.gui != None):
            SliderDisplay.gui.repaint()
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))

    # Use this to change the value (does clamping)
    def set_value(self, value_f):
        value_tick = self.ticks * (value_f - self.low) / self.range
        value_tick = min( max(0, value_tick), self.ticks)
        self.slider.setValue( int(value_tick) )
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))

# The main class for handling the robot drawing and geometry
class DrawRobot(QWidget):
    def __init__(self, gui):
        super().__init__()

        # In order to get to the slider values
        self.gui = gui

        self.title = "Robot arm"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 500
        self.height = 500

        # For doing dictionaries
        self.components = ['upperarm', 'forearm', 'wrist', 'finger1', 'finger2']
        # Set geometry
        self.initUI()

    def initUI(self):
        self.text = "Not reaching"
        self.setWindowTitle(self.title)
        self.setGeometry( self.left, self.top, self.width, self.height )
        self.show()

    # For making sure the window shows up the right size
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    # For making sure the window shows up the right size
    def sizeHint(self):
        return QSize(self.width, self.height)

    # What to draw
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawText(event, qp)
        self.drawTarget(qp)
        self.draw_arm(qp)
        qp.end()

    # Put some text in the bottom left
    def drawText(self, event, qp):
        qp.setPen( QColor(168,34,3) )
        qp.setFont( QFont('Decorative', 10))
        qp.drawText(event.rect(), Qt.AlignBottom, self.text)

    # Map from [0,1]x[0,1] to the width and height of the window
    def x_map(self, x):
        return int(x * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    # Draw a + where the target is and where the end effector is
    def drawTarget(self, qp):
        pen = QPen(Qt.darkGreen, 2, Qt.SolidLine)
        qp.setPen(pen)
        x_i = self.x_map(self.gui.reach_x.value())
        y_i = self.y_map(self.gui.reach_y.value())
        qp.drawLine( x_i-5, y_i, x_i+5, y_i)
        qp.drawLine( x_i, y_i-5, x_i, y_i+5)

        pt = self.arm_end_pt()
        pen.setColor(Qt.darkRed)
        qp.setPen(pen)

        x_i = self.x_map(pt[0])
        y_i = self.y_map(pt[1])
        qp.drawLine( x_i-5, y_i, x_i+5, y_i)
        qp.drawLine( x_i, y_i-5, x_i, y_i+5)

    # Make a rectangle with the center at the middle of the left hand edge
    # Width is 1/4 length
    # returns four corners with points as row vectors
    def make_rect(self, len):
        x_l = 0
        x_r = len
        h = len/4
        y_b = -h/2
        y_t = y_b + h
        return [[x_l,y_b,1], [x_r,y_b,1], [x_r,y_t,1], [x_l,y_t,1] ]

    # Apply the matrix m to the points in rect
    def transform_rect(self, rect, m):
        rect_t = []
        for p in rect:
            p_new = m @ np.transpose(p)
            rect_t.append( np.transpose(p_new) )
        return rect_t

    # Create a rotation matrix
    def rotation_matrix(self, theta):
        m_rot = np.identity(3)
        m_rot[0][0] = cos(theta)
        m_rot[0][1] = -sin(theta)
        m_rot[1][0] = sin(theta)
        m_rot[1][1] = cos(theta)
        return m_rot

    # Create a translation matrix
    def translation_matrix(self, dx, dy):
        m_trans = np.identity(3)
        m_trans[0,2] = dx
        m_trans[1,2] = dy
        return m_trans

    # Draw the given box
    def draw_rect(self, rect, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        for i in range(0,len(rect)):
            i_next = (i+1)%len(rect)
            x_i = self.x_map(rect[i][0])
            y_i = self.y_map(rect[i][1])
            x_i_next = self.x_map(rect[i_next][0])
            y_i_next = self.y_map(rect[i_next][1])
            qp.drawLine( x_i, y_i, x_i_next, y_i_next)

    # Return the matrices that move each of the components. Do this as a dictionary, just to be clean
    def get_matrics(self):
        # The values use to build the matrices
        len_upper_arm = self.gui.length_upper_arm.value()
        len_forearm = self.gui.length_lower_arm.value()
        len_wrist = self.gui.length_lower_arm.value() / 4
        len_finger = self.gui.length_fingers.value()
        h_forearm = len_forearm/4
        ang_shoulder = self.gui.theta_base.value()
        ang_elbow = self.gui.theta_elbow.value()
        ang_wrist = self.gui.theta_wrist.value()
        ang_finger = self.gui.theta_fingers.value()

        mat_ret = dict()

        # begin homework 1 : Problem 2
        # Each of these should be of the form: Translation * rotation
        lens = [len_upper_arm, len_forearm, len_wrist, len_finger, len_finger]
        angs = [ang_shoulder, ang_elbow, ang_wrist, -ang_finger, ang_finger]
        mat_ret['base'] = self.translation_matrix(0, 0.5)
        for i,c in enumerate(self.components):
            h = 0
            if c == 'finger1':
                h = h_forearm*0.75
            elif c == 'finger2':
                h = -h_forearm * 0.75

            mat_ret[c + '_T'] = self.translation_matrix(lens[i],h)
            mat_ret[c + '_R'] = self.rotation_matrix(angs[i])

        # end homework 1 : Problem 2
        return mat_ret

    def draw_arm(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        # Create a rectangle for each component then move it to the correct place then draw it
        rects = dict()
        rects['upperarm'] = self.make_rect( self.gui.length_upper_arm.value() )
        rects['forearm'] = self.make_rect( self.gui.length_lower_arm.value() )
        rects['wrist'] = self.make_rect( self.gui.length_lower_arm.value() / 4 )
        rects['finger1'] = self.make_rect( self.gui.length_fingers.value() )
        rects['finger2'] = self.make_rect( self.gui.length_fingers.value() )
        h_wrist = 0.75 * self.gui.length_lower_arm.value()/4

        # begin homework 1 : Problem 2
        # Transform and draw each component using the matrices in self.get_matrices()
        # Example call:
        #   rect_transform = self.transform_rect( rects['base'], mat )
        #   self.draw_rect( rect_transform, qp )

        matrices = self.get_matrics()
        mat_accum = matrices['base']
        for i,c in enumerate( self.components[0:3] ):
            #   getting the translation matrix for upper arm: matrices['upperarm' + '_T']
            mat_accum = mat_accum @ matrices[c + '_R']
            rect_transform = self.transform_rect( rects[c], mat_accum )
            self.draw_rect( rect_transform, qp )
            mat_accum = mat_accum @ matrices[c + '_T']

        mat_offset = np.identity(3)
        for c in self.components[3:5]:
            if c == 'finger1':
                mat_offset = self.translation_matrix(0, h_wrist)
            elif c == 'finger2':
                mat_offset = self.translation_matrix(0, -h_wrist)

            mat = mat_accum @ mat_offset @ matrices[c + '_R']
            rect_transform = self.transform_rect(rects[c], mat)
            self.draw_rect(rect_transform, qp)
        # end homework 1 : Problem 2

    def arm_end_pt(self):
        """ Return the end point of the arm"""
        matrices = self.get_matrics()
        mat_accum = np.identity(3)
        # begin homework 1 : Problem 3
        mat_accum = matrices['base']
        for c in self.components[0:3]:
            mat_accum = mat_accum @ matrices[c + '_R']
            mat_accum = mat_accum @ matrices[c + '_T']

        mat_accum = mat_accum @ matrices['wrist' + '_T']
        # end homework 1 : Problem 3
        pt_end = mat_accum[0:2,2]
        return pt_end


class RobotArmGUI(QMainWindow):
    def __init__(self,app):
        QMainWindow.__init__(self)
        self.setWindowTitle('ROB 514 2D robot arm')

        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Different do reach commands
        reach_gradient_button = QPushButton('Reach gradient')
        reach_gradient_button.clicked.connect(self.reach_gradient)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        reaches = QGroupBox('Reaches')
        reaches_layout = QVBoxLayout()
        reaches_layout.addWidget( reach_gradient_button )
        reaches_layout.addWidget( reach_jacobian_button )
        reaches.setLayout( reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters = QGroupBox('Arm parameters')
        parameter_layout = QVBoxLayout()
        self.theta_base = SliderDisplay('Angle base', -np.pi/2, np.pi/2, 0)
        self.theta_elbow = SliderDisplay('Angle elbow', -np.pi/2, np.pi/2, 0)
        self.theta_wrist = SliderDisplay('Angle wrist', -np.pi/2, np.pi/2, 0)
        self.theta_fingers = SliderDisplay('Angle fingers', -np.pi/4, 0, -np.pi/8)
        self.length_upper_arm = SliderDisplay('Length upper arm', 0.2, 0.4, 0.3)
        self.length_lower_arm = SliderDisplay('Length lower arm', 0.1, 0.2, 0.15)
        self.length_fingers = SliderDisplay('Length fingers', 0.05, 0.1, 0.075)
        self.theta_slds = []
        self.theta_slds.append( self.theta_base )
        self.theta_slds.append( self.theta_elbow )
        self.theta_slds.append( self.theta_wrist )

        parameter_layout.addWidget(self.theta_base)
        parameter_layout.addWidget(self.theta_elbow)
        parameter_layout.addWidget(self.theta_wrist)
        parameter_layout.addWidget(self.theta_fingers)
        parameter_layout.addWidget(self.length_upper_arm)
        parameter_layout.addWidget(self.length_lower_arm)
        parameter_layout.addWidget(self.length_fingers)

        parameters.setLayout(parameter_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QVBoxLayout()
        self.reach_x = SliderDisplay('x', 0, 1, 0.5)
        self.reach_y = SliderDisplay('y', 0, 1, 0.5)
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.random_reach)
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(random_button)
        reach_point.setLayout(reach_point_layout)

        # The display for the graph
        self.robot_arm = DrawRobot(self)

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addStretch()
        left_side_layout.addWidget(parameters)

        right_side_layout.addWidget( self.robot_arm )
        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderDisplay.gui = self

    # generate a random reach point
    def random_reach(self):
        self.reach_x.set_value( random() )
        self.reach_y.set_value( random() )
        self.robot_arm.repaint()

    def reach_gradient(self):
        """Align the robot end point (palm) to the target point using gradient descent"""

        # Use the text field to say what happened
        self.robot_arm.text = "Not improved"

        # begin homework 2 : Problem 1
        b_improved = False
        d_scl = 0.1
        d_eps = pi/10000
        # Keep trying smaller increments while nothing improves
        while d_scl > 0.0001 and b_improved == False:
            # calculate the current distance
            pt = self.robot_arm.arm_end_pt()
            dist = pow( pt[0] - self.reach_x.value(), 2) + pow( pt[1] - self.reach_y.value(), 2)
            # Try each angle in turn
            for ang in self.theta_slds:
                save_ang = ang.value()

                # Gradient
                ang.set_value( save_ang - d_eps )
                pt_new = self.robot_arm.arm_end_pt()
                dist_new = pow( pt_new[0] - self.reach_x.value(), 2) + pow( pt_new[1] - self.reach_y.value(), 2)

                ang_try = save_ang + d_scl * pi
                if (dist_new < dist):
                    ang_try = save_ang - 0.99 * d_scl * pi

                ang.set_value( ang_try )
                pt_new = self.robot_arm.arm_end_pt()
                dist_new = pow( pt_new[0] - self.reach_x.value(), 2) + pow( pt_new[1] - self.reach_y.value(), 2)
                if (dist_new < dist):
                    b_improved = True
                    dist = dist_new
                    self.robot_arm.text = "Improved {} eps {}".format(ang.name, d_scl)
                else:
                    ang.set_value( save_ang )
            d_scl = d_scl / 2
        #end homework 2 : Problem 1

    def reach_jacobian(self):
        """ Use the Jacobian to calculate the desired angle change"""

        # An example problem of an arm with radius 3 currently at angle theta
        radius = 3
        theta = 0.2
        # Vector to the end point
        r = [ radius * cos(theta), radius * sin(theta), 0]
        # Spin around z
        omega_hat = [0,0,1]
        # always 0 in 3rd component
        omega_cross_r = np.cross(omega_hat, r)
        # Desired x,y change
        dx_dy = np.zeros([2,1])
        dx_dy[0,0] = -0.01
        dx_dy[1,0] = -0.1
        # Jacobian
        J = np.zeros([2,1])
        J[0:2,0] = np.transpose( omega_cross_r[0:2] )
        # Solve
        d_ang = np.linalg.lstsq( J, dx_dy, rcond = None )[0]
        # Check result of solve - should be the same as dx_dy
        res = J @ d_ang
        # The actual point you end up at if you change the angle by that much
        pt_new = [ radius * cos(theta + d_ang), radius * sin(theta + d_ang)]

        # begin homework 2 : Problem 2
        mats = self.robot_arm.get_matrics()
        jacob = np.zeros([2,3])

        matrix_order = ['wrist', 'forearm', 'upperarm']
        mat_accum = np.identity(3)
        for i,c in enumerate(matrix_order):
            mat_accum = mats[c + '_R'] @ mats[c + '_T'] @ mat_accum
            r = [ mat_accum[0,2], mat_accum[1,2], 0 ]
            omega_cross_r = np.cross( omega_hat, r )
            jacob[0:2,2-i] = np.transpose( omega_cross_r[0:2] )

        # Desired change in x,y
        pt_reach = self.robot_arm.arm_end_pt()
        dx_dy[0,0] = self.reach_x.value() - pt_reach[0]
        dx_dy[1,0] = self.reach_y.value() - pt_reach[1]


        # Use pseudo inverse to solve
        d_ang = np.linalg.lstsq( jacob, dx_dy, rcond = None )[0]
        res = jacob @ d_ang
#        print(jacob)
        d_ang_save = [ self.theta_slds[i].value() for i in range(0,3) ]
        d_min = 0
        v_min = pow(dx_dy[0,0], 2) + pow(dx_dy[1,0], 2)
        d_max = min(1, pi / max(d_ang))
        for i,ang in enumerate(d_ang):
            self.theta_slds[i].set_value( self.theta_slds[i].value() + ang )
        pt_reach_move = self.robot_arm.arm_end_pt()
        v_max = pow( pt_reach_move[0] - self.reach_x.value(), 2 ) + pow( pt_reach_move[1] - self.reach_y.value(), 2 )
        d_try = d_min
        v_try = v_min
        if (v_max < v_min):
            d_try = d_max
            v_try = v_max
        while ( d_max - d_min > 0.00001 and v_try > 0.01 ):
            d_try = 0.5 * (d_max + d_min)
            for i,ang in enumerate(d_ang):
                self.theta_slds[i].set_value( d_ang_save[i] + ang * d_try )
            pt_reach_try = self.robot_arm.arm_end_pt()
            v_try = pow( pt_reach_try[0] - self.reach_x.value(), 2 ) + pow( pt_reach_try[1] - self.reach_y.value(), 2 )

            if ( v_try < v_min ):
                v_min = v_try
                d_min = d_try
            elif ( v_try < v_max ):
                v_max = v_try
                d_max = d_try
            elif ( v_max > v_min ):
                v_max = v_try
                d_max = d_try
            else:
                v_min = v_try
                d_min = d_try

        if ( v_min < v_try and v_min < v_max ):
            d_try = d_min

        if ( v_max < v_try and v_max < v_min ):
            d_try = d_max
        
        d_try = 0.2
        for i,ang in enumerate(d_ang):
            self.theta_slds[i].set_value( d_ang_save[i] + ang * d_try )

        pt_reach_res = self.robot_arm.arm_end_pt()
        desired_text = "Desired dx dy {0:0.4f},{1:0.4f},".format(dx_dy[0,0], dx_dy[1,0])
        got_text = " got {0:0.4f},{1:0.4f}".format(res[0,0], res[1,0])
        actual_text = ", actual {0:0.4f},{1:0.4f}".format(pt_reach_res[0], pt_reach_res[1])
        self.robot_arm.text = desired_text + got_text + actual_text
        # to set text
        # self.robot_arm.text = text
        # end homework 2 problem 2

    def draw(self, data):
        self.robot_arm.draw()


if __name__ == '__main__':
    app = QApplication([])

    gui = RobotArmGUI(app)

    gui.show()

    app.exec_()