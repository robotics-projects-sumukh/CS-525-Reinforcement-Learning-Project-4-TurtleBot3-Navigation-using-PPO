import ast  # To safely evaluate the dictionary-like strings

def calculate_goal_reached_percentage(log_file_path):
    # Initialize counters
    total_episodes = 0
    goal_reached_count = 0

    try:
        # Open the log file for reading
        with open(log_file_path, 'r') as file:
            for line in file:
                # Each line is a dictionary-like string with single quotes
                try:
                    # Convert the line from a string with single quotes to a dictionary
                    ep_data = ast.literal_eval(line.strip())
                    total_episodes += 1
                    # Check if 'goal_reached' is True
                    if ep_data.get('goal_reached', False):
                        goal_reached_count += 1
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping invalid line in log file: {e}")

        # Calculate percentage
        if total_episodes == 0:
            print("No episodes found in the log file.")
        else:
            goal_reached_percentage = (goal_reached_count / total_episodes) * 100
            print(f"Percentage of episodes where goal was reached: {goal_reached_percentage:.2f}%")

    except FileNotFoundError:
        print(f"Log file '{log_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
log_file_path = 'models/test.log'  # Replace with your log file path
calculate_goal_reached_percentage(log_file_path)
