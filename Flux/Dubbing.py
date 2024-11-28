from transformers import pipeline
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import edit_distance
from sacrebleu import corpus_chrf


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Initialize translation pipelines with language specifications
mbart_model = "facebook/mbart-large-50-many-to-many-mmt"
opus_mt_model_es = "Helsinki-NLP/opus-mt-en-es"  # English to Spanish
opus_mt_model_it = "Helsinki-NLP/opus-mt-en-it"  # English to Italian
m2m_model = "facebook/m2m100_418M"


# Create translation pipelines
mbart_pipeline_es = pipeline("translation", model=mbart_model, tokenizer=mbart_model, src_lang="en_XX", tgt_lang="es_XX")
mbart_pipeline_it = pipeline("translation", model=mbart_model, tokenizer=mbart_model, src_lang="en_XX", tgt_lang="it_XX")
opus_mt_pipeline_es = pipeline("translation", model=opus_mt_model_es, tokenizer=opus_mt_model_es)
opus_mt_pipeline_it = pipeline("translation", model=opus_mt_model_it, tokenizer=opus_mt_model_it)
m2m_pipeline_es = pipeline("translation", model=m2m_model, tokenizer=m2m_model, src_lang="en", tgt_lang="es")
m2m_pipeline_it = pipeline("translation", model=m2m_model, tokenizer=m2m_model, src_lang="en", tgt_lang="it")

# Function to translate text
def translate(text, model_pipeline):
    return model_pipeline(text)[0]['translation_text']

#script_text = """It's amazing. Who's the artist?
script_text = [
    "Greetings, sir.",
    "He is Mr. Venkat Avadhani.",
    "Special advisor to the Nizam.",
    "Governor Scott is not in India.",
    "State the purpose of your visit to Mr. Edward.",
    "When Governor Scott visited Adilabad recently,",
    "he brought back with him a little girl.",
    "Our Nawab has sent me to tell you something about it.",
    "It's regarding the girl we brought from the Deccan.",
    "It is our police department's opinion that the girl be returned to her village.",
    "It is our Nawab's opinion too-",
    "Why so?",
    "It is a Gond child that you've brought, sir.",
    "So?",
    "So what?",
    "Do they have two horns on their head?",
    "They are an innocent people, sir.",
    "Even if you oppress them they won't raise their voice.",
    "But they have a trait.",
    "They like staying in herds like sheep.",
    "Even if one lamb goes missing it causes them great distress.",
    "This is why the herd has a shepherd.",
    "He protects the herd with his life.",
    "So this tribal is going to sh**t down the mighty British empire with his bow and arrows.",
    "So is this tribal going to take us down?",
    "Don't misunderstand me.",
    "I'm only trying to tell you about him.",
    "The shepherd will travel however far to retrieve the missing lamb.",
    "Be it morning or night,",
    "sun or rain, rocks, mountains, valleys, peaks...",
    "he will search anywhere and everywhere for the missing lamb and will eventually find it.",
    "If at that time the lamb is in the tiger's mouth",
    "he will break its teeth, pry its jaws open",
    "and take the lamb back to its herd.",
    "It seems that the shepherd has come to Delhi to begin his hunt.",
    "FOREST OUTSIDE DELHI",
    "The trap is set. Alert our brother.",
    "Jangu, it's not a wolf. It's a tiger.",
    "Jangu!",
    "I am using you for my need.",
    "Forgive me, brother.",
    "It's been 6 months since we've come to Delhi.",
    "We are doing everything we can to find Malli.",
    "Do you think Malli is alive?",
    "Akthar.",
    "Where did you disappear off to?",
    "There's a lot of work. Come on.",
    "Give it to me.",
    "Hey! You!",
    "Greetings, sir.",
    "The engine died on me again what did you repair?",
    "I've been kicking and kicking",
    "and this damn thing won't start.",
    "Let me check, sir.",
    "It's on reserve, sir.",
    "I'm sorry.",
    "What's the bloody trick?",
    "I didn't do anything, sir.",
    "You removed something earlier and attached it now",
    "so that you can charge me again?",
    "I didn't do anything, sir.",
    "You cheating bastard!",
    "There's no fault of mine, sir.",
    "Sir. Please forgive him, sir.",
    "I didn't do anything, sir.",
    "Forgive him, sir.",
    "Robert, please stop it.",
    "It's not his fault.",
    "This will never happen again, sir! I beg you.",
    "Oh God!",
    "Son.",
    "Careful, son.",
    "Brother, are you okay?",
    "He is a monster.",
    "Look how badly he has beaten him.",
    "Why are you hiding your anger, son?",
    "If they come to know the truth about me...",
    "they will punish you all for giving me shelter.",
    "Even though I wasn't born to you,",
    "you are protecting me with your life.",
    "I shouldn't cause you any harm.",
    "Come what may, I'll never let anyone know my true identity.",
    "And that is the gist of it.",
    "We have a hunter targeting the Governor roaming free in Delhi.",
    "We shouldn't really be bothering about imbecile tribals.",
    "However, our good friend the Nizam",
    "who knows the prowess of these tribals seems to think so.",
    "And since this is a matter regarding the Governor",
    "we should act on it.",
    "And with a good deal of bother.",
    "Alright, sir.",
    "We will apprehend this bugger.",
    "Though I would rather roast this swine on a bed of coals.",
    "Let us have the file, sir.",
    "Well yes, that is the catch, officer.",
    "We have nothing on him.",
    "You mean nothing?",
    "Identifying features?",
    "Criminal history?",
]



# Perform translations
translations = {
    "mBART": {
        "es": translate(script_text, mbart_pipeline_es),
        "it": translate(script_text, mbart_pipeline_it)
    },
    "Opus-MT-ES": {
        "es": translate(script_text, opus_mt_pipeline_es),
        "it": translate(script_text, opus_mt_pipeline_it)
    },
    "M2M-100": {
        "es": translate(script_text, m2m_pipeline_es),
        "it": translate(script_text, m2m_pipeline_it)
    }
}

# Reference translations (You would replace these with actual human translations)
reference_es = ["Saludos, señor. Él es el Sr. Venkat Avadhani. Asesor especial del Nizam."]
reference_it = ["Saluti, signore. Lui è il Sig. Venkat Avadhani. Consigliere speciale del Nizam."]

# Calculate BLEU, TER, and ChrF scores
scores = {}
for model, trans in translations.items():
    scores[model] = {
        "BLEU_ES": sentence_bleu(reference_es, trans["es"].split()),
        "BLEU_IT": sentence_bleu(reference_it, trans["it"].split()),
        "TER_ES": edit_distance(reference_es[0], trans["es"]),  # Using edit_distance as a substitute for TER
        "TER_IT": edit_distance(reference_it[0], trans["it"]),
        "ChrF_ES": corpus_chrf([trans["es"]], [[reference_es[0]]]).score,
        "ChrF_IT": corpus_chrf([trans["it"]], [[reference_it[0]]]).score,
    }

# Display scores
for model, metrics in scores.items():
    print(f"Scores for {model}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

# Ranking models using Mean Win Rate
win_count = {model: 0 for model in translations.keys()}
for metric in ['BLEU_ES', 'BLEU_IT', 'TER_ES', 'TER_IT', 'ChrF_ES', 'ChrF_IT']:
    best_model = min(scores, key=lambda x: scores[x][metric] if "TER" in metric else -scores[x][metric])
    win_count[best_model] += 1

# Calculate Mean Win Rate
mean_win_rate = {model: count / len(translations) for model, count in win_count.items()}

# Print Mean Win Rates
print("\nMean Win Rates:")
for model, win_rate in mean_win_rate.items():
    print(f"{model}: {win_rate:.4f}")

# Final ranking
ranked_models = sorted(mean_win_rate.items(), key=lambda x: x[1], reverse=True)
print("\nRanked Models:")
for model, rate in ranked_models:
    print(f"{model}: {rate:.4f}")
