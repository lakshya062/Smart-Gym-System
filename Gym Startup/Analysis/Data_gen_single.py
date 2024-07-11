import random
import csv
from datetime import datetime, timedelta

# Define exercise schedule by days
exercise_schedule = {
    "Monday": "Bicep Curls"
}

# Define days of the week
days_of_week = ["Monday"]

# Starting and ending angles for a rep
starting_angle_range = (35, 55)
ending_angle_range = (150, 170)

# Define initial weight for bicep curls
initial_weight = 5

def generate_user_exercise_data(user_name, start_date, num_weeks):
    user_data = []
    current_date = start_date
    current_weight = initial_weight
    same_weight_weeks = 0
    for _ in range(num_weeks):
        day_of_week = days_of_week[current_date.weekday()]
        exercise_name = exercise_schedule.get(day_of_week)
        if exercise_name:
            sets = random.randint(2, 3)
            for set_number in range(1, sets + 1):
                reps = random.randint(10, 12)
                for rep_number in range(1, reps + 1):
                    if set_number == sets and sets > 1:
                        starting_angle = round(random.uniform(starting_angle_range[0], starting_angle_range[1] + 15), 1)
                        ending_angle = round(random.uniform(ending_angle_range[0] - 15, ending_angle_range[1]), 1)
                    else:
                        starting_angle = round(random.uniform(starting_angle_range[0], starting_angle_range[1]), 1)
                        ending_angle = round(random.uniform(ending_angle_range[0], ending_angle_range[1]), 1)
                    exercise = {
                        "user_name": user_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "day_of_week": day_of_week,
                        "exercise_name": exercise_name,
                        "weight_kg": current_weight,
                        "starting_angle": starting_angle,
                        "ending_angle": ending_angle,
                        "set_number": set_number,
                        "rep_number": rep_number
                    }
                    user_data.append(exercise)

        if same_weight_weeks < 2 and random.random() < 0.7:
            same_weight_weeks += 1
        else:
            same_weight_weeks = 0
            current_weight += 2.5  # Increase weight by 2.5 kg
        current_date += timedelta(weeks=1)  # Advance by a week
    return user_data

def save_user_data_to_csv(user_data, user_name):
    csv_filename = f"{user_name}_exercise_data.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["user_name", "date", "day_of_week", "exercise_name", "weight_kg", "starting_angle", "ending_angle", "set_number", "rep_number"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for exercise in user_data:
            writer.writerow(exercise)
    print(f"Data for {user_name} saved to {csv_filename}")

if __name__ == "__main__":
    users = ["John", "Alice", "Bob"]
    start_date = datetime.strptime("2024-04-01", "%Y-%m-%d")
    num_weeks = 30
    for user in users:
        user_exercise_data = generate_user_exercise_data(user, start_date, num_weeks)
        save_user_data_to_csv(user_exercise_data, user)
