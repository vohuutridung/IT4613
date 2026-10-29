import type { HistoryEntry } from "../types";

const categories: Record<string, string> = {
  autos: "Autos",
  entertainment: "Entertainment",
  finance: "Finance",
  foodanddrink: "Food & Drink",
  health: "Health",
  kids: "Kids",
  lifestyle: "Lifestyle",
  movies: "Movies",
  music: "Music",
  news: "News",
  sports: "Sports",
  travel: "Travel",
  tv: "TV",
  video: "Video",
  weather: "Weather",
};

const topics: Record<string, string> = {
  autosclassics: "Classic Cars",
  autosenthusiasts: "Car Culture",
  autosluxury: "Luxury Cars",
  autosmotorcycles: "Motorcycles",
  autosnews: "Auto News",
  autosownership: "Car Ownership",
  autosresearch: "Car Research",
  autosresearchguides: "Car Buying Guides",
  autosreview: "Car Reviews",
  autossports: "Sports Cars",
  autossuvs: "SUVs",
  autostrucks: "Trucks",
  baseball_mlb: "MLB Baseball",
  basketball_nba: "NBA Basketball",
  basketball_nba_videos: "NBA Videos",
  basketball_ncaa: "College Basketball",
  football_ncaa: "College Football",
  football_ncaa_videos: "College Football Videos",
  football_nfl: "NFL Football",
  football_nfl_videos: "NFL Videos",
  icehockey_nhl: "NHL Hockey",
  soccer_fifa_wwc: "Women's World Cup",
  more_sports: "More Sports",
  mma: "MMA",
  "entertainment-celebrity": "Celebrity",
  "finance-career": "Careers",
  "finance-career-education": "Careers & Education",
  "finance-companies": "Companies",
  "finance-credit": "Credit",
  "finance-healthcare": "Health Care",
  "finance-insurance": "Insurance",
  "finance-real-estate": "Real Estate",
  "finance-retirement": "Retirement",
  "finance-savemoney": "Saving Money",
  "finance-saving-investing": "Saving & Investing",
  "finance-small-business": "Small Business",
  financenews: "Finance News",
  personalfinance: "Personal Finance",
  foodnews: "Food News",
  foodrecipes: "Recipes",
  quickandeasy: "Quick & Easy",
  restaurantsandnews: "Restaurants",
  "health-news": "Health News",
  healthnews: "Health News",
  mentalhealth: "Mental Health",
  sexualhealth: "Sexual Health",
  weightloss: "Weight Loss",
  "lifestyle-news-feature": "Lifestyle Features",
  lifestylebeauty: "Beauty",
  lifestylebuzz: "Culture",
  lifestylecareer: "Careers",
  lifestylecelebstyle: "Celebrity Style",
  lifestyledidyouknow: "Did You Know?",
  lifestylefamily: "Family",
  lifestylefamilyandrelationships: "Family & Relationships",
  lifestylefashion: "Fashion",
  lifestylehomeandgarden: "Home & Garden",
  lifestylehoroscope: "Horoscopes",
  lifestylelovesex: "Love & Relationships",
  lifestylemarriage: "Marriage",
  lifestylemindandsoul: "Mind & Soul",
  lifestyleparenting: "Parenting",
  lifestylepetsanimals: "Pets & Animals",
  lifestyleroyals: "Royals",
  lifestylesmartliving: "Smart Living",
  lifestylestyle: "Style",
  lifestylevideo: "Lifestyle Videos",
  lifestyleweddings: "Weddings",
  movienews: "Movie News",
  "movies-awards": "Movie Awards",
  "movies-celebrity": "Movie Stars",
  "movies-gallery": "Movies",
  "music-awards": "Music Awards",
  "music-celebrity": "Music Stars",
  "music-gallery": "Music",
  musicnews: "Music News",
  musicvideos: "Music Videos",
  newscrime: "Crime",
  newsfactcheck: "Fact Check",
  newsgoodnews: "Good News",
  newsoffbeat: "Offbeat",
  newsopinion: "Opinion",
  newsphotos: "Photo Stories",
  newspolitics: "Politics",
  newsscienceandtechnology: "Science & Technology",
  newstrends: "In the News",
  newsus: "U.S. News",
  newsweather: "Weather",
  newsworld: "World News",
  popculture: "Pop Culture",
  "shop-all": "Shopping",
  tipsandtricks: "Tips & Tricks",
  topnews: "Top News",
  travelarticle: "Travel Stories",
  travelnews: "Travel News",
  traveltips: "Travel Tips",
  traveltripideas: "Trip Ideas",
  "tv-celebrity": "TV Stars",
  "tv-gallery": "TV",
  "tv-recaps": "TV Recaps",
  tvnews: "TV News",
  weathertopstories: "Weather Stories",
};

const titleCase = (value: string) =>
  value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());

export function categoryLabel(category: string): string {
  return categories[category] ?? titleCase(category);
}

export function topicLabel(topic: string, category?: string): string {
  if (topics[topic]) return topics[topic];
  if (categories[topic]) return categories[topic];
  // Unknown future identifiers still lose a recognized joined category prefix.
  const prefix = Object.keys(categories)
    .sort((a, b) => b.length - a.length)
    .find((name) => topic.startsWith(name) && topic.length > name.length);
  const readable = titleCase(
    prefix ? topic.slice(prefix.length).replace(/^[-_]/, "") : topic,
  );
  return readable || (category ? categoryLabel(category) : "Stories");
}

export const formatCategory = categoryLabel;
export const formatTopic = topicLabel;

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0)
    return "Not recorded";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return minutes
    ? `${minutes} min${remainder ? ` ${remainder} sec` : ""}`
    : `${remainder} sec`;
}

const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** Read source/local ISO components directly; these archives have no timezone metadata. */
function calendarParts(value: string) {
  const iso = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}):\d{2})?$/.exec(
    value,
  );
  if (iso)
    return {
      year: +iso[1],
      month: +iso[2],
      day: +iso[3],
      hour: +(iso[4] ?? 0),
      minute: +(iso[5] ?? 0),
    };
  const source =
    /^(\d{1,2})\/(\d{1,2})\/(\d{4}) (\d{1,2}):(\d{2}):\d{2} (AM|PM)$/.exec(
      value,
    );
  if (!source) return null;
  return {
    year: +source[3],
    month: +source[1],
    day: +source[2],
    hour: (+source[4] % 12) + (source[6] === "PM" ? 12 : 0),
    minute: +source[5],
  };
}

export function formatArchiveDate(
  value: string,
  options: { short?: boolean } = {},
): string {
  const parts = calendarParts(value);
  if (!parts || !months[parts.month - 1]) return "Date unavailable";
  const month = months[parts.month - 1];
  return `${options.short ? month.slice(0, 3) : month} ${parts.day}, ${parts.year}`;
}

export const formatDate = formatArchiveDate;

export function formatArchiveTime(value: string): string {
  const parts = calendarParts(value);
  if (!parts) return "Time unavailable";
  return `${parts.hour % 12 || 12}:${String(parts.minute).padStart(2, "0")} ${parts.hour >= 12 ? "PM" : "AM"}`;
}

export function sortHistory<T extends HistoryEntry>(
  history: readonly T[],
): T[] {
  return [...history].sort((a, b) =>
    b.exposureSortKey.localeCompare(a.exposureSortKey),
  );
}
