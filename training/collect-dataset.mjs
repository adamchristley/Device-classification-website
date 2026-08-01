import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const CONFIG_PATH = path.join(ROOT, 'data', 'dataset_config.json');
const MANIFEST_PATH = path.join(ROOT, 'data', 'manifest.csv');
const API = 'https://commons.wikimedia.org/w/api.php';
const USER_AGENT = 'DeviceArchitectureResearch/0.1 (https://github.com/adamchristley/Device-classification-website)';
const BAD_TITLE_PARTS = [
  'logo', 'icon', 'diagram', 'schematic', 'symbol', 'map', 'patent', 'manual page',
  'screenshot', 'screen shot', 'drawing', 'illustration', 'advertisement', 'poster',
  'packaging', 'box art', 'stamp', 'coin', 'banknote', 'museum label'
];

function csvEscape(value) {
  const text = value == null ? '' : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function plain(metadata, key) {
  return metadata?.[key]?.value?.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() ?? '';
}

function isUsable(page, config) {
  const info = page.imageinfo?.[0];
  if (!info || !info.thumburl || !info.descriptionurl || !info.sha1) return false;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(info.mime)) return false;
  if (Math.min(Number(info.width) || 0, Number(info.height) || 0) < config.minimum_dimension) return false;
  const title = String(page.title || '').toLowerCase();
  if (BAD_TITLE_PARTS.some((part) => title.includes(part))) return false;
  const license = plain(info.extmetadata, 'LicenseShortName');
  return Boolean(license);
}

async function commonsSearch(query, limit, config) {
  const params = new URLSearchParams({
    action: 'query',
    format: 'json',
    formatversion: '2',
    generator: 'search',
    gsrsearch: query,
    gsrnamespace: '6',
    gsrlimit: String(Math.max(30, limit * 6)),
    prop: 'imageinfo',
    iiprop: 'url|mime|size|sha1|extmetadata',
    iiurlwidth: String(config.thumbnail_width),
  });
  const response = await fetch(`${API}?${params}`, { headers: { 'User-Agent': USER_AGENT } });
  if (!response.ok) throw new Error(`Commons API returned ${response.status} for ${query}`);
  const payload = await response.json();
  return (payload.query?.pages ?? []).filter((page) => isUsable(page, config));
}

async function main() {
  const config = JSON.parse(await fs.readFile(CONFIG_PATH, 'utf8'));
  const rows = [];
  const usedSha1 = new Set();

  for (const [label, families] of Object.entries(config.classes)) {
    for (const family of families) {
      process.stdout.write(`Collecting ${label}/${family.family}... `);
      const candidates = await commonsSearch(family.query, config.images_per_family, config);
      let added = 0;
      for (const page of candidates.sort((a, b) => String(a.title).localeCompare(String(b.title)))) {
        const info = page.imageinfo[0];
        if (usedSha1.has(info.sha1)) continue;
        usedSha1.add(info.sha1);
        rows.push({
          dataset_version: config.dataset_version,
          label,
          family: family.family,
          split: family.split,
          source: 'Wikimedia Commons',
          file_title: page.title,
          source_url: info.descriptionurl,
          image_url: info.thumburl,
          original_url: info.url,
          sha1: info.sha1,
          mime: info.mime,
          width: info.width,
          height: info.height,
          license: plain(info.extmetadata, 'LicenseShortName'),
          license_url: plain(info.extmetadata, 'LicenseUrl'),
          artist: plain(info.extmetadata, 'Artist'),
          credit: plain(info.extmetadata, 'Credit'),
          query: family.query,
          review_status: 'weak-label-needs-review',
        });
        added += 1;
        if (added >= config.images_per_family) break;
      }
      console.log(`${added} images`);
      if (added < Math.max(4, Math.floor(config.images_per_family / 2))) {
        throw new Error(`Too few usable images for ${label}/${family.family}: ${added}`);
      }
    }
  }

  const headers = Object.keys(rows[0]);
  const csv = [headers.join(','), ...rows.map((row) => headers.map((key) => csvEscape(row[key])).join(','))].join('\n') + '\n';
  await fs.mkdir(path.dirname(MANIFEST_PATH), { recursive: true });
  await fs.writeFile(MANIFEST_PATH, csv);

  const summary = rows.reduce((acc, row) => {
    const key = `${row.split}:${row.label}`;
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
  console.log(`Wrote ${rows.length} rows to ${path.relative(ROOT, MANIFEST_PATH)}`);
  console.table(summary);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
