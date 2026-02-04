# 📱 Telegram Bot Setup Guide

Get instant trading signal notifications on your phone!

---

## 🎯 Why Telegram?

- ✅ **Instant notifications** - Signals appear on your phone immediately
- ✅ **Mobile-friendly** - Trade from anywhere
- ✅ **Free** - No costs
- ✅ **Secure** - End-to-end encryption
- ✅ **Interactive** - Send commands to your bot

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Your Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather
3. Send: `/newbot`
4. Choose a name: `My Trading Bot`
5. Choose a username: `my_trading_signals_bot` (must end with 'bot')
6. **Copy the token** you receive (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID

1. Search for `@userinfobot` on Telegram
2. Start a chat
3. Send any message
4. **Copy your Chat ID** (looks like: `123456789`)

### Step 3: Configure the Bot

Create a file `.telegram_config.json` in your Test directory:

```json
{
  "token": "YOUR_BOT_TOKEN_HERE",
  "chat_id": "YOUR_CHAT_ID_HERE"
}
```

**Or** set environment variables:

```bash
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN_HERE"
export TELEGRAM_CHAT_ID="YOUR_CHAT_ID_HERE"
```

### Step 4: Test the Bot

```bash
cd ~/Test
python3 src/trading/telegram_bot.py
```

You should receive a test message in Telegram! 🎉

---

## 📋 Bot Commands

Once your bot is running, you can send these commands:

| Command | Description |
|---------|-------------|
| `/start` | Activate bot and see help |
| `/status` | View active positions |
| `/signals` | See recent signals |
| `/stats` | Trading statistics |
| `/help` | Show commands |

---

## 📊 What You'll Receive

### 1. Signal Alerts

When a new trading opportunity is detected:

```
🟢 NEW TRADING SIGNAL 🟢

📊 Signal: RSI_Oversold_5m_Reversal
💰 Instrument: BTCUSDT.P (5m)
📈 Direction: LONG

💵 ENTRY:
   Price: $45,000.00

🛡️ RISK MANAGEMENT:
   Stop Loss: $44,500.00 (-1.11%)
   Take Profit: $46,000.00 (+2.22%)
   R:R Ratio: 1:2.0

💰 POSITION:
   Size: $1,000.00
   Risk: $100.00
   Potential: $200.00

📊 QUALITY:
   Confidence: 65% ⭐⭐⭐
   Backtest WR: 62%
   Backtest PF: 1.5x

⏰ 2026-02-03 14:30:00 UTC
```

### 2. Position Updates

When positions close:

```
✅ POSITION UPDATE ✅

🆔 Position: RSI_5m_001
📊 Status: TARGET_HIT

💰 P&L:
   USD: +$200.00
   Return: +2.0%

⏰ 2026-02-03 15:45:00 UTC
```

### 3. Daily Summaries

End-of-day performance:

```
📊 DAILY SUMMARY 📊

📅 Date: 2026-02-03

✅ PERFORMANCE:
   Total P&L: +$450.00
   Return: +4.5%

📈 TRADES:
   Total: 10
   Wins: 6 ✅
   Losses: 4 ❌
   Win Rate: 60%

Keep up the disciplined trading! 🚀
```

### 4. Alerts

System notifications:

```
⚠️ DAILY LOSS LIMIT ALERT ⚠️

You've hit -3% daily loss limit.
Trading paused for today.
Review your strategy.

⏰ 2026-02-03 16:00:00 UTC
```

---

## 🔒 Security Tips

1. **Keep your token private** - It's like a password
2. **Don't share your bot** - Each trader should have their own
3. **Use 2FA on Telegram** - Extra security for your account
4. **Keep chat_id private** - It's your personal ID

---

## 🛠️ Troubleshooting

### Bot not sending messages?

**Check 1:** Token and Chat ID are correct
```bash
# Test the bot
python3 src/trading/telegram_bot.py
```

**Check 2:** You started a conversation with your bot
- Search for your bot by username
- Click "Start" button

**Check 3:** Network connection
```bash
# Check if you can reach Telegram
ping api.telegram.org
```

### Getting "Unauthorized" error?

- Double-check your bot token
- Make sure you copied the entire token (no spaces)
- Try creating a new bot and token

### Not receiving messages?

- Make sure you clicked "Start" on your bot
- Check your Telegram notifications are enabled
- Try sending `/start` command to your bot manually

---

## 🎨 Customization

### Change Message Format

Edit `src/trading/telegram_bot.py`:

```python
# Customize the signal alert message
async def send_signal_alert(self, signal):
    message = f"""
YOUR CUSTOM MESSAGE HERE
{signal.signal_name}
{signal.entry_price}
"""
```

### Add Custom Commands

```python
async def cmd_mycustom(self, update: Update, context):
    """Your custom command"""
    await update.message.reply_text("Custom response!")

# Register it
self.app.add_handler(CommandHandler("mycustom", self.cmd_mycustom))
```

---

## 📱 Mobile Notifications

### iOS

1. Open Telegram app settings
2. Go to "Notifications and Sounds"
3. Enable "Alerts" for your bot
4. Choose a unique sound for trading signals

### Android

1. Open Telegram settings
2. Go to "Notifications"
3. Set custom notification for your bot
4. Enable vibration/LED for important alerts

---

## 🔄 Integration with Live Scanner

The Telegram bot automatically integrates with the real-time signal scanner:

```python
# In your scanning loop
from src.trading.telegram_bot import TradingTelegramBot

# Initialize
bot = TradingTelegramBot(token, chat_id, config)

# When signal is generated
await bot.send_signal_alert(signal)
```

---

## 💡 Pro Tips

1. **Test first** - Always test your bot before going live
2. **Multiple bots** - Create separate bots for paper vs live trading
3. **Notification schedule** - Configure quiet hours if needed
4. **Group chats** - Add bot to a group for team trading
5. **Backup** - Save your token in a secure password manager

---

## 📚 Advanced Features

### Send Charts

```python
# Send chart image with signal
await bot.bot.send_photo(
    chat_id=bot.chat_id,
    photo=open('chart.png', 'rb'),
    caption='Signal chart'
)
```

### Interactive Keyboards

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

keyboard = [
    [InlineKeyboardButton("Take Trade", callback_data='take')],
    [InlineKeyboardButton("Skip", callback_data='skip')]
]
reply_markup = InlineKeyboardMarkup(keyboard)

await bot.bot.send_message(
    chat_id=bot.chat_id,
    text="New signal!",
    reply_markup=reply_markup
)
```

---

## 🚦 Next Steps

1. ✅ Complete this setup
2. ✅ Test the bot with `python3 src/trading/telegram_bot.py`
3. ✅ Integrate with your signal scanner
4. ✅ Start receiving signals!

---

## 📞 Getting Help

If you run into issues:

1. Check the [Telegram Bot API docs](https://core.telegram.org/bots/api)
2. Test with the bot test script
3. Review error logs
4. Verify your credentials

---

**You're now ready to receive trading signals on your phone!** 📱🚀

Remember: Telegram notifications are instant, so you'll never miss an opportunity.
