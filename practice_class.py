class Car :
    def __init__(self,name ,year, patronous):
        if not name :
            raise ValueError("Missing Name !! Try re-run the program . ")
        if year not in [2024,2025,2026] :
            raise ValueError("Old Model !! Unsafe to Puchase")
        print("Initiaizing object : A new car....")
        self.name = name
        self.year = year
        self.patronous = patronous

    def __str__(self):
        return f"{self.name} from {self.year} so it's a {self.patronous}"

def main():
    car = get_car_info()
    print(car)

def get_car_info() :
    name = input("Name: ")
    year = int(input("Year: "))
    patronous = input("Patronous: ")
    return Car(name ,year,patronous)

if __name__ == "__main__" :
    main()