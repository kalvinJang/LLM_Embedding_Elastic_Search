# 토큰 설정
import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, ConversationHandler
import datetime
import numpy as np

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import pickle
with open('data_0930.pkl', 'rb') as f:
    df1 = pickle.load(f)
import pandas as pd
df = df1[['review', 'no_tag_review']].copy()
df['human_label'] = pd.Series()

test = df['no_tag_review'].tolist()
index = 0
TOKEN = 'token'
REVIEW, NEXT = range(2)

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    # user = update.effective_user
    # await update.message.reply_html(
    #     rf"Hi {user.mention_html()}!",
    #     reply_markup=ForceReply(selective=True),
    # )
    global index
    index = 0
    await update.message.reply_text(test[index])
    return REVIEW

async def handle_response(update, context):
    global index
    response = update.message.text
    print(response, type(response))
    if response == '1':
        # 같은 반응에 대해 같은 값 넣기
        df.loc[df['no_tag_review'] == test[index], 'human_label'] = 1
        index += 1
    elif response == '0':
        df.loc[df['no_tag_review'] == test[index], 'human_label'] = 0
        index += 1
    elif response == 'save':
        with open('tele_revised' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pkl', 'wb') as g:
            pickle.dump(df, g)
    elif response == 'stop':
        with open('tele_revised' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pkl', 'wb') as g:
            pickle.dump(df, g)
        return ConversationHandler.END
    else:
        await update.message.reply_text("잘못된 입력입니다. 다음 문제로 넘어갑니다.")
        index +=1

    try:
        ## 이미 tag가 있는 애들은 나오지 않게하는 코드. strip을 안 해줘서 띄어쓰기가 다르면 나옴
        while df.loc[df['no_tag_review'] == test[index], 'human_label'].sum()>0:
            index +=1
        await update.message.reply_text(test[index])
    except:
        with open('tele_revised' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pkl', 'wb') as g:
            pickle.dump(df, g)
        await update.message.reply_text(test[index])

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            REVIEW: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_response)],
        },
        fallbacks=[]
    )

    # on different commands - answer in Telegram
    application.add_handler(conv_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()