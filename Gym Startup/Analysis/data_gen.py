import random
import csv
from datetime import datetime, timedelta

# Define exercise weights (min, max)
exercise_weights = {
    "Squats": (0, 100),
    "Bench Press": (5, 150),
    "Deadlifts": (10, 200),
    "Pull-Ups": (0, 50),
    "Planks": (0, 5),
}

# Define exercise schedule by days
exercise_schedule = {
    "Monday": {"Bench Press": (5, 150), "Squats": (0, 100)},
    "Tuesday": {"Deadlifts": (10, 200), "Pull-Ups": (0, 50)},
    "Wednesday": {"Squats": (0, 100), "Planks": (0, 5)},
    "Thursday": {"Bench Press": (5, 150), "Pull-Ups": (0, 50)},
    "Friday": {"Deadlifts": (10, 200), "Squats": (0, 100)},
    "Saturday": {"Pull-Ups": (0, 50), "Planks": (0, 5)},
    "Sunday": "Rest",
}

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def generate_user_exercise_data(user_name, start_date, num_days):
    user_data = []
    current_date = start_date
    for _ in range(num_days):
        day_of_week = days_of_week[current_date.weekday()]
        if day_of_week != "Sunday":
            for exercise_name, weight_range in exercise_schedule[day_of_week].items():
                weight_min, weight_max = weight_range
                exercise = {
                    "user_name": user_name,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "day_of_week": day_of_week,
                    "exercise_name": exercise_name,
                    "weight_kg": round(random.uniform(weight_min, weight_max), 2),
                    "sets": random.randint(2, 5),
                    "reps": random.randint(5, 15),
                }
                user_data.append(exercise)
        current_date += timedelta(days=1)
    return user_data

def save_user_data_to_csv(user_data, user_name):
    csv_filename = f"{user_name}_exercise_data.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["user_name", "date", "day_of_week", "exercise_name", "weight_kg", "sets", "reps"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for exercise in user_data:
            writer.writerow(exercise)
    print(f"Data for {user_name} saved to {csv_filename}")

if __name__ == "__main__":
    users = ["John", "Alice", "Bob"]
    start_date = datetime.strptime("2024-04-01", "%Y-%m-%d")
    num_days = 50
    for user in users:
        user_exercise_data = generate_user_exercise_data(user, start_date, num_days)
        save_user_data_to_csv(user_exercise_data, user)




'''
1) fatigue point calculation 
2) Weight increase prediction based on fatigue point calculation
3) diet model
'''
