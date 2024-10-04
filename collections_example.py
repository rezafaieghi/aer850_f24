# List data type
fruits = ["apple", "banana", "orange"]
print("Fruits:", fruits)
print("Data type:", type(fruits))

# Tuple data type
coordinates = (10, 20)
print("Coordinates:", coordinates)
print("Data type:", type(coordinates))

# Set data type
unique_numbers = {1, 2, 3, 4, 5}
print("Unique Numbers:", unique_numbers)
print("Data type:", type(unique_numbers))

# Dictionary data type
student = {"name": "John", "age": 20, "grade": "A"}
print("Student:", student)
print("Data type:", type(student))

class Coordinate3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
aPoint = Coordinate3D(x=1.0, y=2.0, z=3.0)



