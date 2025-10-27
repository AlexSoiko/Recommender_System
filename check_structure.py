import os

def check_project_structure():
    base_path = "/app"
    required_dirs = ['api', 'models', 'data']
    required_files = ['app.py', 'config.py', 'tensorflow_model.py']

    print("🔍 Проверка структуры проекта в Docker контейнере:")

    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            print(f"Папка {dir_name} найдена")
            files = os.listdir(dir_path)
            print(f" Файлы: {files}")
        else:
            print(f"Папка {dir_name} не найдена")

    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            print(f"Файл {file_name} найден")
        else:
            print(f"Файл {file_name} не найден")


if __name__ == "__main__":
    check_project_structure()