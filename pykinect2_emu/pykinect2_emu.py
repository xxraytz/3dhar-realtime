import cv2 as cv2


class ColorFrameDescription:
    def __init__(self):
        self.Width = 1920
        self.Height = 1080


class Position:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ScreenJoint:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __str__(self):
        return f'{self.x} {self.y}'


class Joint:

    def __init__(self, x, y, z, state):
        self.Position = Position(x / 1000, y / 1000, z / 1000)
        self.TrackingState = int(state)

    def __str__(self):
        return f'{self.Position.x} {self.Position.y} {self.Position.z} {self.TrackingState}'


class Bodies:
    def __init__(self, bodies):
        self.bodies = bodies


class Body:
    def __init__(self, joints):
        self.joints = joints
        self.is_tracked = True


class KinectEmulator:

    def __init__(self, path):
        self.__path_to_data = path if path is not None else "data/.."
        self.__body_data = self.__path_to_data + "/Body/FileSkeleton.txt"
        self.__color_data = self.__path_to_data + '/Color/person_stream.mp4'
        self.__frame_number = 0
        self.color_frame_desc = ColorFrameDescription()
        self.max_body_count = 6
        self.__exceed_points = [7, 11, 15, 19, 21, 22, 23, 24]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __get_body_frames(self):
        with open(self.__body_data, 'r+') as file:
            skeleton_frames = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = []
                    for i in range(17):
                        skeleton.append(
                            Joint(*[
                                float(el) for el in file.readline().split(' ')[:4]
                            ])
                        )
                    for i in self.__exceed_points:
                        skeleton.insert(i, Joint(0, 0, 0, 0))
                    skeleton_frames.append(skeleton)
        return skeleton_frames

    def __get_color_body_frames(self):
        with open(self.__body_data, 'r+') as file:
            skeleton_frames = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = []
                    for i in range(17):
                        skeleton.append(
                            ScreenJoint(*[
                                float(el) for el in file.readline().split(' ')[4:]
                            ])
                        )
                    for i in self.__exceed_points:
                        skeleton.insert(i, ScreenJoint(0, 0))
                    skeleton_frames.append(skeleton)
        return skeleton_frames

    def __body_frames_generator(self):
        skeleton_frames = self.__get_body_frames()
        while True:
            yield [
                Body(skeleton_frames[self.__frame_number])
            ] * self.max_body_count

    def __color_body_frames_generator(self):
        skeleton_frames = self.__get_color_body_frames()
        while True:
            yield skeleton_frames[self.__frame_number]

    def __color_frames_generator(self):
        cap = cv2.VideoCapture(self.__color_data)
        while True:
            ret, frame = cap.read()
            if ret:
                self.__frame_number += 1
                yield frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.__frame_number = 0
                continue
        cap.release()

    @staticmethod
    def has_new_body_frame():
        return True

    def get_last_body_frame(self):
        return Bodies(next(self.__body_frames_generator()))

    def get_last_color_frame(self):
        return next(self.__color_frames_generator())

    @staticmethod
    def has_new_color_frame():
        return True

    def body_joints_to_color_space(self, joints):
        return next(self.__color_body_frames_generator())
