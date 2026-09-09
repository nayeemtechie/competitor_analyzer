import assert from 'node:assert/strict';
import test from 'node:test';

import { analyzeHtml, buildResult } from '../lib/audit.ts';

const productCard = (name: string) => `
  <article class="product-card" data-product-id="${name}">
    <a href="/products/${name}"><img src="/${name}.jpg" alt="${name}">${name}</a>
    <span>$29.00</span>
  </article>`;

void test('detects and classifies a popular-products widget from DOM evidence', () => {
  const html = `<html><body>
    <section class="recommendations product-carousel">
      <h2>Popular right now</h2>
      ${productCard('alpha')}${productCard('beta')}${productCard('gamma')}
    </section>
  </body></html>`;

  const result = analyzeHtml(html, 'https://shop.example/collections/new');
  assert.equal(result.pageType, 'category');
  assert.ok(result.placements.length >= 1);
  assert.equal(result.placements[0].detectedType, 'Popular');
  assert.equal(result.placements[0].productCount, 3);
  assert.ok(result.placements[0].evidence.some((item) => item.includes('heading')));
});

void test('detects a frequently-bought-together placement on a product page', () => {
  const html = `<html><body itemscope itemtype="https://schema.org/Product">
    <main><h1>Primary product</h1></main>
    <aside data-widget="frequently-bought-together">
      <h3>Frequently bought together</h3>
      ${productCard('case')}${productCard('charger')}
    </aside>
  </body></html>`;

  const result = analyzeHtml(html, 'https://shop.example/products/phone');
  assert.equal(result.pageType, 'product');
  assert.ok(result.placements.some((item) => item.detectedType === 'Frequently Bought Together'));
});

void test('does not treat an ordinary navigation grid as a recommendation placement', () => {
  const html = `<html><body><nav><h2>Shop departments</h2>
    <a href="/collections/women"><img src="/women.jpg">Women</a>
    <a href="/collections/men"><img src="/men.jpg">Men</a>
  </nav></body></html>`;

  assert.deepEqual(analyzeHtml(html, 'https://shop.example/').placements, []);
});

void test('builds coverage, scores, and evidence-backed opportunities', () => {
  const popular = analyzeHtml(`<section class="recommendations"><h2>Trending</h2>${productCard('a')}${productCard('b')}</section>`, 'https://shop.example/');
  const product = analyzeHtml('<html><body itemscope itemtype="https://schema.org/Product"><h1>Item</h1></body></html>', 'https://shop.example/products/item');
  const result = buildResult('https://shop.example/', [
    { url: 'https://shop.example/', pageType: popular.pageType, placements: popular.placements },
    { url: 'https://shop.example/products/item', pageType: product.pageType, placements: product.placements },
  ]);

  assert.equal(result.pagesAnalyzed.length, 2);
  assert.ok(result.coverage.some((item) => item.pageType === 'product' && item.percentage === 0));
  assert.ok(result.opportunities.every((item) => item.evidenceUrls.length > 0));
  assert.ok(result.overallScore >= 0 && result.overallScore <= 100);
});
