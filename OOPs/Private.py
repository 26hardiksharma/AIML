class BankAccount:
    def __init__(self,name,balance):
        self.name = name
        self.__balance = balance #Private Attribute


b1 = BankAccount("Hardik",10000)

