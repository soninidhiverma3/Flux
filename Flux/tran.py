from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import CHRF
import numpy as np
from transformers import pipeline
# Instantiate the CHRF metric
chrf_metric = CHRF()

# Set up the translation pipelines
translator_tagalog = pipeline('translation', model='Helsinki-NLP/opus-mt-en-tl')  # English to Tagalog
translator_swahili_mbart = pipeline('translation', model='facebook/mbart-large-50-many-to-one-mmt', src_lang="en_XX", tgt_lang="sw_KE")  # mBART for Swahili
translator_m2m = pipeline('translation', model='facebook/m2m100_418M', src_lang="en", tgt_lang="sw")  # M2M-100 for Swahili

# Sample script lines from RRR
script_lines = [
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



reference_translations_tagalog = [
    "Mga pagbati, sir.",
    "Siya ay Mr. Si Venkat Avadhani.",
    "Espesyal na tagapayo ng Nizam.",
    "Ang Gobernador Scott ay wala sa India.",
    "Ipahayag ang layunin ng iyong pagbisita sa Mr. Edward.",
    "Kapag si Gobernador Scott ay bumisita sa Adilabad kamakailan,",
    "nagdala siya ng isang batang babae.",
    "Ang aming Nawab ay nagpadala sa akin upang sabihin sa iyo ang isang bagay tungkol dito.",
    "Ito ay tungkol sa batang babae na dinala namin mula sa Deccan.",
    "Ito ay ang opinyon ng aming departamento ng pulisya na ang batang babae ay ibalik sa kanyang nayon.",
    "Ito rin ang opinyon ng Nawab.",
    "Bakit kaya?",
    "Ito ay isang Gond bata na iyong dinala, sir.",
    "Baka naman?",
    "At ano naman?",
    "May dalawang sungay ba ang mga ito sa kanilang ulo?",
    "Sila ay isang inosenteng mga tao, sir.",
    "Kahit pa'y pinipilit mo sila ay hindi nila itataas ang kanilang tinig.",
    "Ngunit mayroon silang isang katangian.",
    "Gusto nilang manatili sa mga kawan tulad ng mga tupa.",
    "Kahit na ang isang kordero ay nawawala ito ay nagdudulot sa kanila ng malaking kabagabagan.",
    "Ito ang dahilan kung bakit ang kawan ay may pastol.",
    "Pinaprotektahan niya ang kawan sa pamamagitan ng kanyang buhay.",
    "Kaya ang tribal na ito ay pupuksain ang makapangyarihang imperyo ng Britanya sa pamamagitan ng kanyang busog at mga pana.",
    "Kaya ba'y pupuksain tayo ng tribal na ito?",
    "Huwag kang magkamali.",
    "Sinusubukan ko lamang na sabihin sa iyo tungkol sa kanya.",
    "Ang pastol ay maglalakbay kahit na gaano kalayuan upang kunin ang nawawalang kordero.",
    "Maging umaga man o gabi,",
    "Araw o ulan, mga bato, mga bundok, mga libis, mga taluktok...",
    "hahanapin niya ang nawawalang kordero sa lahat ng dako at sa huli ay makikita niya ito.",
    "Kung sa oras na iyon ang kordero ay nasa bibig ng tigre,",
    "puputulin niya ang mga ngipin nito, bubuksan ang mga panga nito,",
    "at ibabalik ang kordero sa kawan nito.",
    "Mukhang ang pastol ay dumating sa Delhi upang simulan ang kanyang pangangaso.",
    "LANGIT sa labas ng Delhi.",
    "Ang silo ay nakahanda na. Ipaalam sa ating kapatid.",
    "Jangu, hindi ito lobo. Ito ay isang tigre.",
    "Jangu!",
    "Ginagamit kita para sa aking pangangailangan.",
    "Patawarin mo ako, kapatid.",
    "Ito ay 6 buwan mula nang dumating kami sa Delhi.",
    "Ginagawa namin ang lahat upang mahanap si Malli.",
    "Sa palagay mo ba'y buhay si Malli?",
    "Akthar.",
    "Nasaan ka nawala off sa?",
    "Maraming trabaho. Sige na.",
    "Ibigay mo sa akin.",
    "Hoy! Ikaw!",
    "Mga pagbati, sir.",
    "Ang makina ay namatay sa akin muli kung ano ang naayos mo?",
    "Ako'y tumatakbo at tumatakbo,",
    "at hindi magsisimulang magsimula ang kasuklam-suklam na ito.",
    "Hayaan mo akong suriin, sir.",
    "Ito ay sa reserba, sir.",
    "Pasensya na.",
    "Ano ba ang daming trick?",
    "Wala akong ginawa, sir.",
    "Ikaw ay inalis ang isang bagay na mas maaga at naka-attach ito ngayon,",
    "upang maaari mo akong parusahan muli?",
    "Wala akong ginawa, sir.",
    "Ikaw na pandaraya na bastardo!",
    "Walang kasalanan ko, sir.",
    "Panginoon. Pakiusap patawarin mo siya, sir.",
    "Wala akong ginawa, sir.",
    "Patawarin mo siya, sir.",
    "Robert, ihinto mo na.",
    "Hindi siya ang may kasalanan.",
    "Hindi na ito maulit, sir! Humihingi ako sa iyo.",
    "Oh Diyos ko!",
    "Anak.",
    "Mag-ingat ka, anak.",
    "Kapatid, okay ka ba?",
    "Siya ay isang halimaw.",
    "Tingnan mo kung gaano siya ka-badtrip.",
    "Bakit mo tinatago ang iyong galit, anak?",
    "Kung malalaman nila ang katotohanan tungkol sa akin...",
    "Sila'y parurusahan kayong lahat sa pagbibigay sa akin ng kanlungan.",
    "Kahit na hindi ako ipinanganak sa iyo,",
    "Ikaw ay pinoprotektahan ako sa iyong buhay.",
    "Hindi ko dapat saktan ka.",
    "Kung ano man ang mangyari, hinding-hindi ko ipapaalam sa sinuman ang tunay kong pagkatao.",
    "At iyan ang kabuuan nito.",
    "May isang mangangaso na naglalayong sa Gobernador na malaya sa Delhi.",
    "Hindi natin dapat ipagkakaabalahan ang mga mangmang na tribal.",
    "Gayunman, ang ating mabuting kaibigan na si Nizam,",
    "Sino ang nakakaalam ng katatagan ng mga tribal na ito ay tila nag-iisip ng gayon.",
    "At dahil ito ay isang bagay na may kinalaman sa Gobernador,",
    "Dapat na tayo'y kumilos.",
    "At sa pamamagitan ng isang mahusay na pakikitungo ng pag-aalala.",
    "Okay na, sir.",
    "Halikanin natin ang taong ito.",
    "Bagaman mas gugustuhin kong iprito ang baboy na ito sa isang kama ng mga baga.",
    "Ibigay mo sa amin ang file, sir.",
    "Oo nga, iyan ang problema, opisyal.",
    "Walang alam natin tungkol sa kanya.",
    "Wala ka bang ibig sabihin?",
    "Pagkilala sa mga katangian?",
    "Mga Kasaysayan ng Krimen?"
]


reference_translations_swahili = [
"Salamu, bwana.",
 "Yeye ni Bwana Venkat Avadhani.",
 "Mshauri maalum wa Nizam.",
 "Gavana Scott hayuko India.",
 "Taja madhumuni ya ziara yako kwa Bw. Edward.",
 "Gavana Scott alipotembelea Adilabad hivi majuzi,",
 "alirudi pamoja naye msichana mdogo.",
 "Nawab wetu amenituma nikuambie kitu kuhusu hilo."
 "Ni kuhusu msichana tuliyemleta kutoka Deccan.",
 "Ni maoni ya idara yetu ya polisi kwamba msichana huyo arudishwe kijijini kwake.",
 "Ni maoni ya Nawab wetu pia-",
 "Kwa nini?",
 "Ni mtoto wa Gond ambaye umeleta, bwana.",
 "Kwa hiyo?",
 "Basi nini?",
 "Je! wana pembe mbili juu ya vichwa vyao?"
 "Hao ni watu wasio na hatia, bwana."
 "Hata ukiwadhulumu hawatapaza sauti zao."
 "Lakini wana tabia."
 "Wanapenda kukaa katika makundi kama kondoo."
 "Hata mwana-kondoo mmoja akipotea huwaletea dhiki kubwa."
 "Ndio maana kundi lina mchungaji."
 "Analinda kundi kwa maisha yake.",
 "Kwa hivyo kabila hili litaangamiza ufalme mkubwa wa Uingereza kwa upinde na mishale yake."
 "Kwa hiyo kabila hili litatushusha?",
 "Usinielewe vibaya.",
 "Ninajaribu tu kukuambia juu yake."
 "Mchungaji atasafiri hata umbali gani ili kumchukua mwana-kondoo aliyepotea."
 "Iwe asubuhi au usiku,"
 "jua au mvua, miamba, milima, mabonde, vilele ...",
 "atatafuta popote na kila mahali kwa mwana-kondoo aliyepotea na hatimaye atampata.",
 "Ikiwa wakati huo mwana-kondoo yuko kwenye kinywa cha chui",
 "Atamvunja meno yake, atafungua taya zake",
 "na kumrudisha mwana-kondoo kwenye kundi lake."
 "Inaonekana mchungaji amekuja Delhi kuanza kuwinda.",
 "MSITU NJE YA DELHI",
 "Mtego umewekwa. Tahadharisha ndugu yetu.",
 "Jangu, sio mbwa mwitu. Ni tiger.",
 "Jangu!",
 "Ninakutumia kwa mahitaji yangu.",
 "Nisamehe, kaka."
 "Imekuwa miezi 6 tangu tuje Delhi.",
 "Tunafanya kila tuwezalo kumpata Malli.",
 "Unafikiri Malli yuko hai?",
 "Aktar.",
 "Ulipotea kwenda wapi?",
 "Kuna kazi nyingi. Njoo.",
 "Nipe.",
 "Hey! Wewe!",
 "Salamu, bwana.",
 "Injini ilinifia tena ulitengeneza nini?",
 "Nimekuwa nikipiga teke",
 "na jambo hili mbaya halitaanza.",
 "Hebu niangalie, bwana."
 "Iko kwenye hifadhi, bwana.",
 "Samahani.",
 "Ujanja wa umwagaji damu ni nini?",
 "Sikufanya chochote, bwana."
 "Uliondoa kitu hapo awali na kuambatanisha sasa",
 "ili uweze kunitoza tena?",
 "Sikufanya chochote, bwana."
 "Unadanganya mwanaharamu!",
 "Hakuna kosa langu, bwana.",
 "Bwana. Tafadhali msamehe, bwana."
 "Sikufanya chochote, bwana."
 "Msamehe, bwana.",
 "Robert, tafadhali acha.",
 "Sio kosa lake."
 "Hii haitatokea tena, bwana! Nakuomba.",
 "Ee Mungu!",
 "Mwana.",
 "Makini, mwanangu."
 "Kaka, uko sawa?",
 "Yeye ni monster.",
 "Angalia jinsi alivyompiga vibaya."
 "Kwanini unaficha hasira yako mwanangu?"
 "Ikiwa watakuja kujua ukweli kunihusu ...",
 "watawaadhibu nyote kwa kunipa hifadhi.",
 "Ingawa sikuzaliwa kwako,"
 "Unanilinda na maisha yako."
 "Sipaswi kukusababishia madhara yoyote.",
 "Hata iweje, sitawahi kuruhusu mtu yeyote kujua utambulisho wangu wa kweli.",
 "Na huo ndio msingi wake."
 "Tuna mwindaji anayemlenga Gavana anayezurura bure huko Delhi.",
 "Hatupaswi kuwa na wasiwasi juu ya makabila ya wajinga.",
 "Hata hivyo, rafiki yetu mzuri Nizam",
 "Nani anajua uwezo wa makabila haya inaonekana kufikiria hivyo.",
 "Na kwa kuwa hili ni suala linalomhusu Gavana",
 "tunapaswa kuchukua hatua juu yake."
 "Na kwa shida nyingi.",
 "Sawa bwana.",
 "Tutamkamata mdudu huyu.",
 "Ingawa ni afadhali niwachome nguruwe huyu kwenye kitanda cha makaa."
 "Tupe faili bwana.",
 "Ndio, hiyo ni samaki, afisa.",
 "Hatuna chochote juu yake."
 "Huna maana?",
 "Kutambua sifa?",
 "Historia ya uhalifu?",

]

# Translate script lines to Tagalog and Swahili using the pipelines
translated_lines_tagalog = [translator_tagalog(line)[0]['translation_text'] for line in script_lines]
translated_lines_swahili_mbart = [translator_swahili_mbart(line)[0]['translation_text'] for line in script_lines]
translated_lines_swahili_m2m = [translator_m2m(line)[0]['translation_text'] for line in script_lines]

# Define the smoothing function for BLEU score calculation
smoothing_function = SmoothingFunction().method1
def evaluate_metrics(reference_translations, translated_lines):
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    chrf_scores = []
    
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for ref, translated in zip(reference_translations, translated_lines):
        # BLEU Score
        bleu_score = sentence_bleu([ref.split()], translated.split(), smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)
        
        # ROUGE Score
        rouge_score = rouge_scorer_instance.score(ref, translated)
        rouge1 = rouge_score['rouge1'].fmeasure
        rouge2 = rouge_score['rouge2'].fmeasure
        rougel = rouge_score['rougeL'].fmeasure
        rouge_scores.append((rouge1 + rouge2 + rougel) / 3)  # Average of ROUGE-1, ROUGE-2, and ROUGE-L
        
        # METEOR Score - split the strings into lists of words
        meteor = meteor_score([ref.split()], translated.split())
        meteor_scores.append(meteor)
        
        # CHRF Score
        chrf_score = chrf_metric.sentence_score(translated, [ref]).score
        chrf_scores.append(chrf_score)
    
    # Calculate mean scores
    mean_bleu = np.mean(bleu_scores)
    mean_rouge = np.mean(rouge_scores)
    mean_meteor = np.mean(meteor_scores)
    mean_chrf = np.mean(chrf_scores)
    
    return {
        'BLEU': mean_bleu,
        'ROUGE': mean_rouge,
        'METEOR': mean_meteor,
        'CHRF': mean_chrf
    }


# Evaluate Tagalog translations
metrics_tagalog = evaluate_metrics(reference_translations_tagalog, translated_lines_tagalog)

# Evaluate Swahili translations (mBART)
metrics_swahili_mbart = evaluate_metrics(reference_translations_swahili, translated_lines_swahili_mbart)

# Evaluate Swahili translations (M2M-100)
metrics_swahili_m2m = evaluate_metrics(reference_translations_swahili, translated_lines_swahili_m2m)

# Display the metrics for each model
print("\nEvaluation Metrics for Tagalog LLM:")
print(metrics_tagalog)

print("\nEvaluation Metrics for Swahili mBART LLM:")
print(metrics_swahili_mbart)

print("\nEvaluation Metrics for Swahili M2M-100 LLM:")
print(metrics_swahili_m2m)

# Calculate Mean Win Rate for ranking models
models = ['Tagalog LLM', 'Swahili mBART LLM', 'Swahili M2M-100 LLM']
mean_scores = [np.mean(list(metrics_tagalog.values())),
               np.mean(list(metrics_swahili_mbart.values())),
               np.mean(list(metrics_swahili_m2m.values()))]

# Rank models based on Mean Win Rate (higher mean score indicates better performance)
ranked_models = sorted(zip(models, mean_scores), key=lambda x: x[1], reverse=True)

print("\nModel Rankings Based on Mean Win Rate (Average of all Metrics):")
for rank, (model, score) in enumerate(ranked_models, start=1):
    print(f"{rank}. {model}: Mean Score = {score:.4f}")
