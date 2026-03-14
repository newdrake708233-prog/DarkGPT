import os
import time
import wikipediaapi

#algortihm to pull text from wikipedia for model training data

OUTPUT_FILE = "training.txt"
DELAY       = 0.5

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="dark_gpt_trainer/1.0"
)

#topics to pull from wikipedia for training data
#currently just general history 1900+, and knowledge on python
#can be changed if running locally from github
#will just require model retrain (10000 steps will probably take around 35 depending on gpu)
#Initial train on rx7600, required 30-35 min for 30 mb of training data.

TOPICS = [

    "Python (programming language)",
    "Computer security",
    "Penetration testing",
    "Ethical hacking",
    "Malware",
    "Social engineering (security)",
    "Cybercrime",
    "Encryption",
    "Network security",
    "Exploit (computer security)",
    "SQL injection",
    "Buffer overflow",
    "Reverse engineering",
    "Linux",
    "Command-line interface",
    "Open-source software",
    "World War I",
    "World War II",
    "Causes of World War I",
    "Causes of World War II",
    "Western Front (World War I)",
    "Eastern Front (World War II)",
    "Holocaust",
    "Manhattan Project",
    "Atomic bombings of Hiroshima and Nagasaki",
    "D-Day",
    "Battle of Stalingrad",
    "Treaty of Versailles",
    "Cold War",
    "Cuban Missile Crisis",
    "Korean War",
    "Vietnam War",
    "Space Race",
    "Berlin Wall",
    "Soviet Union",
    "NATO",
    "Marshall Plan",
    "McCarthyism",
    "Nuclear arms race",
    "Russian Revolution",
    "Chinese Communist Revolution",
    "Iranian Revolution",
    "French Resistance",
    "Civil rights movement",
    "Decolonisation",
    "Indian independence movement",
    "Apartheid",
    "Nelson Mandela",
    "Mahatma Gandhi",
    "Suffragette movement",
    "Arab–Israeli conflict",
    "Gulf War",
    "War in Afghanistan (2001–2021)",
    "Iraq War",
    "Syrian civil war",
    "Rwandan genocide",
    "Bosnian War",
    "Falklands War",
    "Troubles (Northern Ireland)",
    "Soviet–Afghan War",
    "Yom Kippur War",
    "Six-Day War",
    "Russian invasion of Ukraine",
    "Israel–Hamas war (2023–present)",
    "China–United States relations",
    "Taiwan Strait",
    "Nord Stream pipeline",
    "2023 Wagner Group mutiny",
    "Russia–NATO relations",
    "Belt and Road Initiative",
    "South China Sea",
    "Iran nuclear deal",
    "Saudi Arabia–Iran relations",
    "2021 Afghan presidential crisis",
    "Political history of the United States",
    "History of Russia",
    "History of China",
    "History of the United Kingdom",
    "History of Germany",
    "History of France",
    "European Union",
    "United Nations",
    "History of Japan",
    "History of India",
    "History of the Middle East",
    "History of Africa",
    "Fascism",
    "Communism",
    "Liberalism",
    "Conservatism",
    "Nationalism",
    "Populism",
    "Totalitarianism",
    "Democracy",
    "Authoritarianism",
    "Socialism",
    "Adolf Hitler",
    "Joseph Stalin",
    "Winston Churchill",
    "Franklin D. Roosevelt",
    "Mao Zedong",
    "Vladimir Lenin",
    "Vladimir Putin",
    "Xi Jinping",
    "Barack Obama",
    "Donald Trump",
    "Angela Merkel",
    "Volodymyr Zelenskyy",
    "September 11 attacks",
    "War on terror",
    "Al-Qaeda",
    "Islamic State of Iraq and the Levant",
    "2005 London bombings",
    "2015 Paris attacks",
    "Global counterterrorism",
    "Great Depression",
    "2008 financial crisis",
    "COVID-19 pandemic",
    "Globalisation",
    "Oil crisis",
    "Bretton Woods system",
    "World Trade Organization",
    "International Monetary Fund",
    "Climate change",
    "Paris Agreement",
    "Gulag",
    "Cultural Revolution",
    "Armenian genocide",
    "Cambodian genocide",
    "Nanjing massacre",
    "Srebrenica massacre",
    "History of the Internet",
    "Social media",
    "Artificial intelligence",
    "Nuclear proliferation",
    "Space exploration",
    "Digital surveillance",
    "Cyberwarfare",

]

def fetch_all():
    total_chars = 0
    failed      = []

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, topic in enumerate(TOPICS, 1):
            try:
                page = wiki.page(topic)

                if not page.exists():
                    print(f"  [{i:3d}/{len(TOPICS)}]Not found: {topic}")
                    failed.append(topic)
                    continue

                text = page.text.strip()
                if len(text) < 500:
                    print(f"  [{i:3d}/{len(TOPICS)}]Too short, skipping: {topic}")
                    failed.append(topic)
                    continue
                f.write(f"TOPIC: {topic}\n")
                f.write(text)
                f.flush()

                total_chars += len(text)
                print(f"  [{i:3d}/{len(TOPICS)}]{topic:<45} {len(text):>10,} chars")

                time.sleep(DELAY)

            except Exception as e:
                print(f"  [{i:3d}/{len(TOPICS)}]Error on '{topic}': {e}")
                failed.append(topic)
                time.sleep(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'─' * 60}")
    print(f"Done!")
    print(f"File:       {output_path}")
    print(f"Size:       {size_mb:.1f} MB")
    print(f"Characters: {total_chars:,}")
    print(f"Articles:   {len(TOPICS) - len(failed)}/{len(TOPICS)} fetched")

    if failed:
        print(f"\nFailed topics ({len(failed)}):")
        for t in failed:
            print(f"   - {t}")

    print(f"\nNow run:  python train.py --data {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        import wikipediaapi
    except ImportError:
        print("❌ Missing dependency. Run:  pip install wikipedia-api")
        exit(1)

    fetch_all()