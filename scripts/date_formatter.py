# Reformat the dates
turkish_months = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
}

def format_month_year(date_str):
    try:
        parts = date_str.split("/")  # expected: "00/MM/YYYY"
        month = int(parts[1])
        year = parts[2]
        month_name = turkish_months[month]
        return f"{month_name} {year}"  # e.g., "Ocak 2024"
    except Exception as e:
        return ""
