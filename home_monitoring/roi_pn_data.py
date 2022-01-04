from os import name
import cv2
import json

coor = []
coor2 = []

class DrawLineWidget(object):
    def __init__(self):
        img = cv2.imread(input_img_dir)
        self.original_image = cv2.resize(img, (1280,720))
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        global coor, coor2
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            cv2.circle(self.clone, self.image_coordinates[0], 4, (0, 0, 255), -1)
            """
            if self.image_coordinates[0][1] > 25:
                cv2.putText(self.clone, str(self.image_coordinates[0][0])+','+str(self.image_coordinates[0][1]), (self.image_coordinates[0][0],self.image_coordinates[0][1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, 11)
            else:
                cv2.putText(self.clone, str(self.image_coordinates[0][0])+','+str(self.image_coordinates[0][1]), (self.image_coordinates[0][0],self.image_coordinates[0][1]+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, 11)
            """
            cv2.imshow("image", self.clone)
        
        elif event == cv2.EVENT_LBUTTONUP:
            coor.append((x,y))

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            #coor = []
            #self.clone = self.original_image.copy()
            for i in range(len(coor)):
                if i == len(coor)-1:
                    cv2.line(self.clone, coor[i], coor[0], (36,255,12), 2)
                else:
                    cv2.line(self.clone, coor[i], coor[i+1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        elif event == cv2.EVENT_RBUTTONUP:
            print(coor)
            coor2 = coor.copy()
            coor = []

    def show_image(self):
        return self.clone

def capture(input_dir):
    vid = cv2.VideoCapture(input_dir)

    _, frame1 = vid.read()
    frame = cv2.resize(frame1, (1280,720))

    cv2.imwrite(input_img_dir, frame)
    print('Capture successfully')

    vid.release()

def finish(output_dir):
    global coor2

    data = {}
    for i in range(len(coor2)):
        data['xY'+str(i)] = coor2[i]

    with open(output_dir, 'w') as outfile:
        json.dump(data, outfile, indent=3)
    outfile.close()
    print("Successfully created", output_dir)

def draw(input_dir):
    capture(input_dir)

    draw_line_widget = DrawLineWidget()

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            finish(json_dir)
            cv2.imwrite(output_img_dir, draw_line_widget.show_image())
            cv2.destroyAllWindows()
            break

def read_roi(input_dir):
    with open(input_dir, 'r') as jsonfile:
        config = json.load(jsonfile)
    roi = []
    for i in range(len(config)):
        roi.append(config['xY'+str(i)])
    return roi

json_dir = "./json/limit.json"
input_img_dir = "./images/input_limit.jpg"
output_img_dir = "./images/output_limit.jpg"
vid_dir = "./videos/"

option = input("    1. Draw\n    2. Read\nType number to choose: ")
if option == "1":
    name = input("Enter video name: ")  # gait3.mp4
    draw(vid_dir + name)
elif option == "2":
    print(read_roi(json_dir))
else:
    exit()