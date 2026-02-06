class Product:
    count = 0
    def __init__(self,name,price):
        self.name = name
        self.price = price
        Product.count+=1
    def get_info(self):
        print(f"Name: {self.name} | Price: {self.price}")
    
    @staticmethod
    def get_count():
        print(Product.count)

p1 = Product("Laptop",10000)
p2 = Product("Washing Machine",20000)

p1.get_info()
p2.get_info()
p1.get_count()