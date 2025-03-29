+++
date = '2025-03-24T09:39:30+01:00'
draft = false
title = 'Cloudflare’s AI Bot Trap'
+++

[Cloudflare](https://www.cloudflare.com/) has introduced an [AI-generated labyrinth](https://blog.cloudflare.com/ai-labyrinth/) to detect and deter unauthorized bots and AI crawlers from scraping content. The "AI Labyrinth" aims to slow down and confuse these bots by leading them through AI-generated pages, wasting their computing power.

Currently, AI crawlers generate over 50 billion requests daily on Cloudflare's network. Instead of blocking them outright, which has led to an arms race, Cloudflare’s approach functions as a honeypot. The system identifies unauthorized crawlers and redirects them into a maze of fake pages. Since human users wouldn’t navigate through multiple misleading links, this behavior helps distinguish bots from real visitors. The gathered data is then used to improve bot detection.