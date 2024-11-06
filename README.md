# HalltapeRoadmapDE
<i>Roadmap для Data Engineer | Актуально на 2024-2025г.</i>
## Стартуем

### 0. Деньги
Размер зарплаты зависит от успешности продажи себя на собесе. Если будешь бояться говорить большие суммы, эти суммы будет называть другой более наглый человек. При этом он будет знать меньше тебя, а зарабатывать больше.
С этого момента забудь про справедливость. Её нет.

Зарплатные вилки собраны лично мной на собесах за 2024 год:
| Уровень      | Зарплата на руки      |
|--------------|--------------------|
| Стажер       | 70k - 90k          |
| Джун         | 100k - 150k        |
| Джун+        | 160k - 190k        |
| Мидл        | 200k - 250k        |
| Мидл+       | 260k - 380k        |
| Сеньор и выше | от 380k        |

Учитывайте, что вилки в некоторых компаниях могут быть сильно ниже. Корреляция между размером зарплаты и знаниями не всегда 100%.

>Важно! Нет смысла заниматься раз в неделю. Таким темпом вы никогда не дойдете до конца. Лучше тогда потратьте это время на отпуск, семью, друзей. Иначе вы просто спускаете свою жизнь в никуда.


***

### 1. Видео
Чтобы понять, кто такой дата инженер и как им стать, посмотри видео!

➜ [Кто такой Data Engineer?](https://youtu.be/75Vu8NqH_cU?si=zYT6U7deVYEPkbmA)


<p align="center">
    <img src="png/de1.jpg" alt="de" width="600"/>
</p>



***
### 2. Курсы
Дальше тебе нужно научиться писать базовые запросы на SQL и на Python. В тренажерах внизу проходи так, как там просят. Не спрашивай зачем, тебе понадобятся эти инструменты в будущем!

Пройди эти три курса SQL и Python

<table>
  <tr>
    <th align="center">Курс</th>
    <th align="center">Продолжительность</th>
  </tr>
  <tr>
    <td align="center"><a href="https://karpov.courses/simulator-sql">SQL с нуля до оконок</a></td>
    <td align="center">≈ 1-2 месяца</td>
  </tr>
  <tr>
    <td align="center"><a href="https://stepik.org/course/58852/syllabus">Python с нуля до базы</a></td>
    <td align="center">≈ 1-2 месяца</td>
  </tr>
  <tr>
    <td align="center"><a href="https://stepik.org/course/68343/syllabus">Продвинутый уровень Python (вложенность, словари, функции)</a></td>
    <td align="center">≈ 1-2 месяца</td>
  </tr>
</table>

>Если задача не получается и ты сидишь с ней уже больше часа, пропускай и переходи к следующей. Потом вернешься и дорешаешь, если будет желание. Не гонись за 100%. Это никто не оценит.
>
***
### 3. Github / Git

<p align="center">
    <img src="png/git_github.png" alt="git" width="600"/>
</p>

Регистрируешься на Github и подключаешь его к своему ПК

➜ [Работа с github / git](Git/README.md)

***
### 4. Linux / Terminal
<p align="center">
    <img src="png/BASH_logo.png" alt="linux" width="600"/>
</p>

Пробуешь привыкнуть и запомнить работу с этими командами в терминале

➜ [Работа с Linux / Terminal](Linux/README.md)

***
### 4. Data Warehouse

<p align="center">
    <img src="png/data_warehouse.png" alt="dwh" width="600"/>
</p>


Нужно понимать, что такое хранилище данных, какие они бывают, чем отличаются и, как в целом можно грузить данные. Обязательно читай теорию!

➜ [Теория по Data Warehouse](DWH/README.md)


***
### 5. Нормальные формы

<p align="center">
    <img src="png/normal_table.jpg" alt="nf" width="600"/>
</p>


Важная тема про нормализацию таблиц. Всегда спрашивают на собесах. За это надо шарить.

➜ [Нормальные формы](NF/README.md)


***
### 6. Модели данных

<p align="center">
    <img src="png/models_data.jpg" alt="nf" width="600"/>
</p>

Для собесов и в будущем на работе вам надо шарить за модели данных. Читаем и обязательно изучаем SCD по ссылке ниже!

➜ [Модели данных](DM/README.md)



***
### 999. Hadoop

<p align="center">
    <img src="png/hadoop_logo.png" alt="hdfs" width="600"/>
</p>


На некоторых проектах в качестве хранилища будет HDFS (Hadoop). Инфы из видоса снизу будет достаточно, чтобы успешно ответить на вопросы на собеседовании.

Смотри видео здесь ➜ [HDFS | Что это такое и как оно работает? [Hadoop HDFS]](https://youtu.be/ySDGh_1d87g)

Презентация из видео ➜ [HDFS]('files/deep_dive_hdfs_pdf.pdf')

***
### 999. Greenplum

<p align="center">
    <img src="./png/gp_logo.png" width="640" height="320"/>
</p>

Greenplum будет в 50% вакансиях на DE. Остальные будут сидеть на Hadoop + Spark. На первых порах рекомендую **базово освоить** все три, но окунуться поглубже лишь в один на выбор (Spark | Greenplum). Если хватит сил на освоение обоих, флаг вам в руки!

 ➜ [Теория по Greenplum](GREENPLUM/README.md)

Презентация по Greenplum ➜ [Greenplum]('files/deep_dive_hdfs_pdf.pdf')

***
### 999. Spark

<p align="center">
    <img src="png/spark_logo.png" alt="spark" width="600"/>
</p>


Spark изучайте только **после** **того**, как научились базово кодить на **python** и **SQL**. Без них будет очень **сложно** и **непонятно**.

Смотри видео здесь ➜ [Что такое Spark и как им пользоваться?](h)


***
### 999. Pet Project

<p align="center">
    <img src="png/pet_project.png" alt="pet_project" width="600"/>
</p>

На пет проект посмотрит 1 из 10 человек, но именно он может быть тем, кто возьмет вас в итоге на работу, поэтому увеличивайте свои шансы. Вам точно должно повезти! Ниже пример моих проектов.

Проекты, с которыми я выходил на свою первую работу:
- [ETL-проект на базе Docker-Compose: Весь стек в одном контейнере](PET_PROJECT/README.md)
- [Telegram бот для генерации паролей](https://github.com/halltape/HalltapePassBot)

Проекты коллег DE, которые я рекомендую посмотреть:
- [Обработка данных SpaceX API](https://github.com/ShustGF/spacex-api-analize)
- [ETL-проект для начинающих Data Engineers: От почтового сервера до Greenplum](https://github.com/dim4eg91/DataEngineering/blob/main/articles/ETL_as_a_pet_project.md)

> Будет круто, если ты напишешь свой собственный проект и запушишь его к себе на github. Это сильно поможет уложить в голове многие концепции при работы с данными

