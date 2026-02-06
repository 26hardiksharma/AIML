class Employee:
    start_time = "9 AM"
    end_time ="5 PM"

class Teacher(Employee):
    def __init__(self,name):
        super().__init__()
        self.name = name

t1 = Teacher("Hardik")
print(t1.start_time,t1.end_time, t1.name)