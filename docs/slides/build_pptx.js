// Wrap the rendered slide images (deck_<lang>.html -> PDF -> PNG) into PPTX files with speaker notes.
const pptxgen = require("pptxgenjs");
const path = require("path");
const DIR = __dirname;
const NOTES = {
  he: [
    "הבעיה ואיך מצאתי אותה. הרצתי את המערכת על 296 שאלות וקראתי את הכשלים. 63 מתוך 187 היו במקום 2 או 3, ברובם מאחורי עמוד אח, כמו דרכון זמני מול ביומטרי. ההשערה: ההבדל נמצא בכותרת. בדיקה בקובץ ההגדרות: מוטמע רק content. 64% מהפסקאות כבר פותחות בכותרת הסעיף, אז מה שחסר הוא כותרת העמוד. נקודת הפתיחה: 37% במקום הראשון, 58% בשלושת הראשונים, כלומר ב-42% מהשאלות GPT לא מקבל את העמוד הנכון.",
    "הפתרון: מפתח אחד בקובץ ההגדרות, והכותרת נכנסת לטקסט שמוטמע. המודל לא אומן מחדש, החיפוש וה-API לא השתנו. מדדתי על שלושה סטים: של Webiks, סט נקי שבניתי שבו הכותרת הוסתרה ממי שכתב את השאלות, והסט הציבורי של מתחרה. על Webiks מ-37% ל-46%, על הנקי מ-53% ל-60%, על הציבורי בתוך הרעש. גרסה עם וקטור כותרת נפרד נתנה 53% על Webiks אבל לא החזיקה על הסטים הנקיים, אז כיביתי אותה לפני ההגשה.",
    "שלב שני אופציונלי, כבוי: מודל שפה מסדר מחדש 30 עמודים מועמדים לפי הכותרות. עם השלב הזה 60% על Webiks ו-67% על הנקי. LLM factory מאפשר כל ספק. הוא כבוי כי הוא מכניס קריאת LLM לחיפוש: כ-0.8 שנייה, מפתח, ואין לו השפעה במצב mock. מה נבדק ונפסל: קטעים קצרים, הקשר שנכתב ב-LLM, ניסוח מחדש של השאלה. 60 בדיקות עוברות, וחוזרים למקור במחיקת מפתח אחד.",
  ],
  en: [
    "How I found the problem. I ran the original system on 296 questions and read the failures: 63 of 187 were at rank 2 or 3, mostly behind a sibling page, like temporary vs biometric passport. Hypothesis: the difference is in the title. The config embeds content only. 64% of paragraphs already start with their section heading, so the page title is what is missing. Starting point: 37% at rank 1, 58% in the top 3, so for 42% of questions GPT never gets the right page.",
    "The fix: one config key puts the title into the embedded text. No retraining; search and API unchanged. Measured on three sets: Webiks' split, a clean set I built with titles hidden from the writer, and a competitor's public set. Webiks 37% to 46%, clean 53% to 60%, public within noise. A variant with a separate title vector reached 53% on the Webiks set but did not hold on the clean sets, so I switched it off before submitting.",
    "Optional second stage, off by default: an LLM reorders 30 candidate pages by title. With it, 60% on Webiks and 67% on the clean set. The LLM factory accepts any provider. It is off because it puts an LLM call in the search path: ~0.8 s, a key, and no effect in mock mode. Rejected: smaller chunks, LLM-written context, query rewriting. 60 tests pass; removing one key restores the original.",
  ],
};
async function build(lang, out) {
  const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; pres.author = "Ariel Halevy";
  pres.title = lang === "he" ? "שיפור האחזור: כותרת העמוד במה שמוטמע" : "Retrieval: the page title in the embedded text";
  for (let i = 1; i <= 3; i++) {
    const s = pres.addSlide();
    s.addImage({ path: path.join(DIR, `hi_${lang}-${i}.png`), x: 0, y: 0, w: 10, h: 5.625 });
    s.addNotes(NOTES[lang][i - 1]);
  }
  await pres.writeFile({ fileName: path.join(DIR, out) });
  console.log("written", out);
}
(async () => { await build("he", "slides_he.pptx"); await build("en", "slides.pptx"); })();
