class Teacher:
    def __init__(self,salary):
        self.salary = salary

class Student:
    def __init__(self,gpa):
        self.gpa =gpa

class TA(Teacher,Student):
    def __init__(self, salary,gpa,name):
        super().__init__(salary)
        Student.__init__(self,gpa)
        self.name = name

t1 = TA("10000",9,"Hardik")

print(t1.name,t1.salary,t1.gpa)
