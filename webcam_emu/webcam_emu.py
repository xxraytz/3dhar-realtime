class ColorFrameDescription:
    def init(self, width=0, height=0):
        self.Width = width
        self.Height = height

class Position:

    def init(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class ScreenJoint:

    def init(self, x, y):
        self.x = x
        self.y = y

class Joint:

    def init(self, x, y, z, state):
        self.Position = Position(x / 1000, y / 1000, z / 1000)
        self.TrackingState = int(state)

    def str(self):
        return f'{self.Position.x} {self.Position.y} {self.Position.z} {self.TrackingState}'

class Bodies:
    def init(self, bodies):
        self.bodies = bodies

class Body:
    def init(self, joints, is_tracked):
        self.joints = joints
        self.is_tracked = is_tracked

class WebcamEmulator:

    def init(self):
        self.color_frame_desc = ColorFrameDescription(width=640, height=480)
        self.max_body_count = 1
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)

    def color_frames_generator(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                yield frame
            else:
                print("Camera not accessible")
                break
        self.cap.release()

    def __body_frames_generator(self):
        for frame in self.__color_frames_generator():
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                joints = [
                    Joint(landmark.x * 1000, landmark.y * 1000, landmark.z * 1000, 2)
                    for landmark in results.pose_landmarks.landmark
                ]
                yield [Body(joints, True)]
            else:
                yield [Body([], False)]

    def get_last_body_frame(self):
        return Bodies(next(self.__body_frames_generator()))

    def get_last_color_frame(self):
        return next(self.__color_frames_generator())

    def has_new_color_frame(self):
        return True

    def has_new_body_frame(self):
        return True

    def body_joints_to_color_space(self, joints):
        return [ScreenJoint(j.Position.x * self.color_frame_desc.Width, j.Position.y * self.color_frame_desc.Height) for j in joints]

    def __enter(self):
        return self

    def exit(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        self.pose.close()