class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0

    def accelerate(self, increment):
        self.speed += increment

    def brake(self, decrement):
        if decrement > self.speed:
            self.speed = 0
        else:
            self.speed -= decrement

    def display_speed(self):
        print("Current speed:", self.speed, "mph")


# Create an instance of the Car class
my_car = Car("Toyota", "Camry", 2021)

# Access and modify attributes
print("Make:", my_car.make)
print("Model:", my_car.model)
print("Year:", my_car.year)

# Call methods to perform actions
my_car.accelerate(30)
my_car.display_speed()

my_car.brake(10)
my_car.display_speed()