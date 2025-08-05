import re

def extract_turkish_date(text):
    aylar = [
        "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
        "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"
    ]
    ay_regex = r"|".join(aylar)

    # Extract the year
    yil_match = re.search(r"\b(20\d{2})\s*['’]?(te|de)?\b", text)
    yil = yil_match.group(1) if yil_match else None

    # Extract the month
    ay_match = re.search(rf"\b({ay_regex})(?:\s*20\d{{2}})?(?:ın|in|un|ün|a|e|da|de|ta|te|nda|nde|’ta|’te|’da|’de| ayında)?\b", text, re.IGNORECASE)
    ay = ay_match.group(1).capitalize() if ay_match else None

    # Check if the date written like this: "Mart2024"
    if not ay or not yil:
        match = re.search(rf"\b({ay_regex})(20\d{{2}})", text, re.IGNORECASE)
        if match:
            ay = match.group(1).capitalize()
            yil = match.group(2)

    # Concat them
    if ay and yil:
        return f"{ay} {yil}"
    elif ay:
        return ay
    elif yil:
        return yil
    else:
        return ""
