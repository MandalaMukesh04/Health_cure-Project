tasks = []

def add_task():
    description = input("Enter task description: ")
    due_date = input("Enter due date (optional): ")
    task = {"description": description, "due_date": due_date, "completed": False}
    tasks.append(task)
    print("Task added successfully.")

def view_tasks():
    if not tasks:
        print("No tasks.")
    else:
        print("To-Do List:")
        for idx, task in enumerate(tasks, 1):
            status = "Completed" if task["completed"] else "Not Completed"
            print(f"{idx}. Description: {task['description']}, Due Date: {task['due_date']}, Status: {status}")

def mark_completed():
    view_tasks()
    choice = int(input("Enter the number of the task you want to mark as completed: "))
    if 1 <= choice <= len(tasks):
        tasks[choice - 1]["completed"] = True
        print("Task marked as completed.")
    else:
        print("Invalid task number.")

def delete_task():
    view_tasks()
    choice = int(input("Enter the number of the task you want to delete: "))
    if 1 <= choice <= len(tasks):
        del tasks[choice - 1]
        print("Task deleted successfully.")
    else:
        print("Invalid task number.")

def main():
    while True:
        print("\nTo-Do List Menu:")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task as Completed")
        print("4. Delete Task")
        print("5. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            add_task()
        elif choice == "2":
            view_tasks()
        elif choice == "3":
            mark_completed()
        elif choice == "4":
            delete_task()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if _name_ == "_main_":
main()