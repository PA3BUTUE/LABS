import telebot
import requests
import json
from telebot import types

bot = telebot.TeleBot("5619854846:AAHSrwpBsdm1cgB3a5RxVt0RtXXsrN59Xjk")
open_weather_token = '222cdd1cb2a4970411e8aa5f8bf5e319'


@bot.message_handler(commands=["start"])
def start(message):
    mess = f"Добрый день, {message.from_user.first_name}, желаете узнать прогноз погоды?"
    bot.send_message(message.chat.id, mess)


@bot.message_handler(content_types=['text'])
def mess_resp(message):
    get_weather(message)


def get_weather(message):
    city = message.text
    r = requests.get(
        url=f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={open_weather_token}&units=metric')

    try:
        data = r.json()
        tmp = data["main"]["temp"]
        tmp_min = data["main"]["temp_min"]
        tmp_max = data["main"]["temp_max"]
        feels_like = data["main"]["feels_like"]
        pressure = data["main"]["pressure"]
        humidity = data["main"]["humidity"]
        wind_sp = data["wind"]["speed"]
        mess = f"Погода в {city} сегодня\nТемпература сейчас: {tmp}°\nМинимальная температура: {tmp_min}°\nМаксимальная температура: {tmp_max}°\nПо ощущениям: {feels_like}°\nДавление: {pressure}\nВлажность: {humidity}\nСкорость ветра: {wind_sp}"
    except KeyError:
        mess = "Ошибка! Введите корректное название города."

    bot.send_message(message.chat.id, mess, parse_mode="html")


bot.polling(none_stop=True)
