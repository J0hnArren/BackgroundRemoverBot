import telebot
from requests.exceptions import ConnectTimeout
from telebot import types
from dotenv import dotenv_values
from model_pytorch import *
from model_tensorflow import *

config = dotenv_values(".env")['TOKEN']
bot = telebot.TeleBot(config, parse_mode=None)


def keyboard():
    btn1 = '/start'
    # btn2 = 'Cut background'
    btn34 = ['FAQ', '/help']
    keyboard_markup = types.ReplyKeyboardMarkup(one_time_keyboard=False, resize_keyboard=True)
    keyboard_markup.add(btn1)
    # keyboard_markup.add(btn2)
    keyboard_markup.add(*btn34)
    return keyboard_markup


@bot.message_handler(commands=['start'])
def send_welcome(message):
    # username
    bot.send_message(message.chat.id, f"Hello, {message.from_user.first_name}! \nSend here a picture in which you "
                                      "want to cut out the background. Clear images of people on a visually "
                                      "well-separated background are processed best.",
                     reply_markup=keyboard())


@bot.message_handler(commands=['help'])
def send_help(message):
    bot.send_message(message.chat.id, "Send here an image which you want to cut out the background.\nClear images of "
                                      "people on a visually well-separated background are processed best.",
                     reply_markup=keyboard())


@bot.message_handler(content_types=["text"])
def show_info(message):
    try:
        if message.text == 'FAQ':
            bot.send_message(message.chat.id, "Author's contacts\nGithub: https://github.com/J0hnArren \nemail: "
            "rauf.parchiev@gmail.com",
                             reply_markup=keyboard())
        else:
            bot.reply_to(message, "Invalid message")
    except ConnectTimeout as e:
        print(e)
        bot.reply_to(message, "Something went wrong with connection. Try again", reply_markup=keyboard())
    except Exception as e:
        print(e)
        bot.reply_to(message, "Something went wrong. Try again", reply_markup=keyboard())


@bot.message_handler(content_types=['photo'])
def handle_picture(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        _, file_extension = os.path.splitext(file_info.file_path)
        filename = message.photo[1].file_id + file_extension

        src = 'data/'
        path = src + filename
        with open(path, 'wb') as new_file:
            new_file.write(downloaded_file)
        print('done')
        bot.send_message(message.chat.id, """Cutting out the background...""",
                         reply_markup=keyboard())

        Model_Tensorflow().cut_image_bg_tf(src, filename)
        print('tf done')
        Model_Pytorch().cut_image_bg_pytorch(path)
        print('torch done')

        bot.send_photo(message.chat.id, open(path, 'rb'), caption="PyTorch model result:", reply_markup=keyboard())
        bot.send_photo(message.chat.id, open(src + "w_bg.png", 'rb'), caption="Tensorflow model result:",
                       reply_markup=keyboard())
        clear_images()

    except ConnectTimeout as e:
        print(e)
        bot.reply_to(message, "Something went wrong with connection. Try again", reply_markup=keyboard())
    except Exception as e:
        print(e)
        bot.reply_to(message, "Something went wrong. Resend the image, please.", reply_markup=keyboard())


def clear_images():
    path = "./data/"
    for file in os.listdir(path):
        if file != 'sample.jpg':
            os.remove(os.path.join(path, file))


def main():
    # bot.polling()
    bot.infinity_polling()
    clear_images()


if __name__ == '__main__':
    main()
