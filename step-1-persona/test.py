class AIAgent:
    def __init__(self, name):
        self.name = name

    def provide_task_details(self, task):
        task_details = {
            "task1": "Task 1 involves data collection and preprocessing.",
            "task2": "Task 2 involves training a machine learning model.",
            "task3": "Task 3 involves evaluating the model performance.",
            "task4": "Task 4 involves deploying the model to production."
        }
        return task_details.get(task, "Task details not found.")

if __name__ == "__main__":
    agent = AIAgent("TaskDetailAgent")
    task = input("Enter the task name (e.g., task1, task2, task3, task4): ")
    details = agent.provide_task_details(task)
    print(details)