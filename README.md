<p align="center"><img src="resources/icons/main.svg"></p>

# Когнитивное моделирование систем управления

Программный комплекс, предназначенный для работы с когнитивными моделями систем управления. Позволяет создавать имитационные данные, оценивать прогноз, находить управляющие воздействия для достижения желаемого состояния системы.

## Запуск приложения
Программы были написаны при использовании Python 3.9
1. Установить требования

`pip install -r requirements.txt`

2. Запустить главный скрипт

`python CM_main.pyw`

## Помощь по использованию
Подробное руководство по использованию приложения находится в файле help.chm

Там же описаны основные термины и определения, используемые в приложении

## Краткое описание структуры проекта
```
examples/                # Примеры
examples/models/         # Примеры когнитивных моделей
examples/data/           # Пример имитационных данных в разных форматах
forms/                   # Содержит скомпилированные ui файлы (классы форм с графическим интерфейсом)
logic/                   # Содержит файлы с основной логикой форм
model/                   # Содержит основную логику когнитивных моделей
model/CM_classes.py      # Содержит классы когнитивных моделей, факторов и межфакторных связей
model/CM_funcs.py        # Содержит функции применения когнитивных моделей для решения задач управления
resources/               # Файлы используемых ресурсов
resources/forms_ui/      # ui файлы с формами Qt
resources/icons/         # Иконки и картинки
help.chm                 # Файл помощи с подробным руководством пользователя 
CM_main.pyw              # Основной файл
forms_resources_rc.py    # Скомпилированный файл ресурсов
```

## Примеры окон приложения
![Alt-текст](screenshot1.jpg)
