""" 
Firebase Connector class
Sochivoath Chiv 2021
"""

import os
import datetime

class IncorrectMode(Exception):
    """Exception for invalid mode setting 
    
    Raises and exception if the mode entered when initializing the FirebaseConnector class is incorrect
    
    Args:
        mode (string): Name of the mode (Camera vs Sensors)
    """
    def __init__(self, mode, message='Invalid Mode entered. Examples include => "Camera" or "Sensor"'):
        self.mode = mode
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return "{} -> {}".format(self.mode, self.message)

class InvalidPerson(Exception):
    """Exception for invalid person
    
    Raises an exception if the the person is not registered in the database.
    
    Args:
        PersonName (string): Name of the person
    """
    def __init__(self, personName, message='Person does not exist in the database. Please use addPerson function to add a new person or check for mispellings (Names are case sensitive).'):
        self.personName = personName
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return "{} -> {}".format(self.personName, self.message)  
    
class FirebaseConnector(object):
    """FireBase Connector class.
    
    Attributes:
        db: Database connection
        root: Root directory
        mode (string): Camera or Sensor 
    """
    
    def __init__(self, db, mode="Camera"):
        self.db = db
        self.root = self.db.reference("/")
        self.mode = mode
        self.check_mode()
    
    def check_mode(self):
        cam = ["c","cam","camera","CAM","Cam","Camera","C","CAMERA"]
        sen = ["s","S","Sen","sen","Sens","sens","sensor","Sensors","sensors","Sensor","SENSORS","SENSOR"]
        
        if self.mode in cam:
            self.mode = "Camera"
        elif self.mode in sen:
            self.mode = "Sensor"
        else:
            raise IncorrectMode(self.mode)
        
    def addPerson(self, personName):
        """Function to add a Person to Firebase
        
        A new person generates a child from the root directory with 2 more children (Camera and Sensor)
        
        Structure:
        
        Root
          |---- personName
                    | ------- Camera
                    | ------- Sensor

        Args:
            personName (String): Name of the person

        Returns:
            boolean: Returns True if added and False if person already exists
        """
        
        if personName not in self.root.get():
            self.root.child(personName).set({
                "Camera": -1,
                "Sensor": -1
                
            })
            print("[INFO] {} Successfully Added".format(personName))
            return True
            
        else:
            print("[INFO] {} already exists. Failed to add person".format(personName))
            return False
    
    def removePerson(self, personName):
        """Remove a person and all of its children from the database.

        Args:
            personName (string): name of the child to be removed
            
        Returns:
            boolean: Returns True if the person was removed successfully
        """
        
        if personName in self.root.get():
            self.root.child(personName).delete()
            print("[INFO] {} Successfully removed".format(personName))
        else:
            raise InvalidPerson(personName)
            
    
    def addAction(self, personName, action):
        curr_day = datetime.date.today()
        date = curr_day.strftime("%d-%m-%Y")
        curr_time = datetime.datetime.now() 
        time = curr_time.strftime("%H:%M:%S")
        
        if personName in self.root.get():
            child = "/"+personName+"/"+self.mode
            ref = self.db.reference(child)
            
            
            if ref.get() == -1:
                ref.child(date).set({
                    time:action
                })
            else:
                if date not in ref.get():
                    ref.child(date).set({
                        time:action
                    })
                else:
                    ref2 = self.db.reference(child+"/" +date)   
                    ref2.child(time).set(action)
            print("Action added")
        else:
            raise InvalidPerson(personName)
        
        
        
        
                
        
                
