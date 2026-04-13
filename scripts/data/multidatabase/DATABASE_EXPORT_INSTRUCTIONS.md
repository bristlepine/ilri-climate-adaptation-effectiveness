# Multi-Database Export Instructions

**Upload all export files to the shared Google Drive folder:**
https://drive.google.com/drive/folders/1f5y8kjVAcHXBm74AM2wOXsdxrCTnh-ll

Place each database's files in its own subfolder:
```
multidatabase/wos/wos_export_1.ris, wos_export_2.ris, ...
multidatabase/cab/cab_export_1.ris, cab_export_2.ris, ...
multidatabase/asp/asp_export_1.ris, asp_export_2.ris, ...
multidatabase/agris/agris_export.ris
```

Search strings are derived from the canonical pipeline query at:
`scripts/outputs/step1/step1_total_query.txt`

---

## Summary

| Database | Access needed | When |
|---|---|---|
| AGRIS | None — fully open | Anytime |
| Web of Science | Cornell current staff/student | On campus or VPN |
| CAB Abstracts | Cornell current staff/student | On campus or VPN |
| Academic Search Premier | Cornell current staff/student | On campus or VPN |

**Log hit counts here as each search is run:**

| Database | Search date | Hit count | Batches exported | Notes |
|---|---|---|---|---|
| AGRIS | | | | |
| Web of Science | | | | |
| CAB Abstracts | | | | |
| Academic Search Premier | | | | |

---

## AGRIS (no login required)

Full string file: `scripts/data/multidatabase/agris/search_string_agris.txt`

AGRIS only supports basic keyword search — no proximity operators. Use the simplified string below:

```
smallholder* OR "small-scale farmer*" OR "subsistence farmer*" OR pastoralist* OR fisher* OR "agroforestry farmer*"
AND
"climate change" OR "climate adaptation" OR drought* OR flood* OR "climate risk" OR "climate variability"
AND
(adapt* OR resilience OR "coping strateg*" OR "adaptive capacity")
AND
(indicator* OR monitor* OR evaluat* OR assess* OR metric*)
AND
(africa OR asia OR "latin america" OR lmic* OR "developing countr*")
```

**Steps:**
1. Go to https://agris.fao.org/agris-search/search.do
2. Use Advanced Search — paste each line into a separate field joined with AND
3. Filter: Publication Year → 2005 to present
4. Note the total hit count in the table above
5. Export: **Export** → **RIS** or **CSV** (exports all results at once)
6. Upload to Drive: `multidatabase/agris/agris_export.ris`

---

## Web of Science (Cornell campus or VPN)

Full string file: `scripts/data/multidatabase/wos/search_string_wos.txt`

Adapted from pipeline query: `TITLE-ABS-KEY` → `TS`, `W/N` → `NEAR/N`, date applied via UI filter.

**Query** (paste into Advanced Search):

```
TS=((((smallholder* OR "small holder*" OR "small-scale farmer*" OR "small-scale producer*" OR "subsistence farmer*" OR "farmers in low-income countr*" OR "small-scale farmers in low-income countr*" OR "marginal farmer*" OR "resource-poor farmer*" OR  ((farmer* OR producer*) NEAR/3 (smallholder* OR "small-scale" OR subsistence OR marginal* OR "resource-poor"))) OR ("rural household*" OR "farm household*" OR "agricultural household*" OR "family farmer*" OR "low-income household*" OR "poor household*" OR "female-headed household*" OR "women-headed household*" OR ((household* OR family OR farmer*) NEAR/3 (rural OR agricultural OR farm* OR "low-income" OR poor OR female-headed OR women-headed))) OR (pastoralist* OR "agro-pastoralist*" OR "agro pastoralist*" OR herder* OR "livestock keeper*" OR "livestock farmer*" OR "livestock producer*" OR "dairy farmer*" OR "crop-livestock farmer*" OR (pastoralist* OR herder* OR ((farmer* OR producer* OR keeper*) NEAR/3 (livestock OR dairy OR cattle OR crop-livestock)) OR "agro-pastoral*")) OR (fisher* OR "small-scale fisher*" OR "small scale fisher*" OR fisherfolk OR "artisanal fisher*" OR "aquaculture producer*" OR "fish farmer*" OR "shrimp farmer*" OR ((fisher* NEAR/3 (small-scale OR smallscale OR artisanal)) OR ((farmer* OR producer*) NEAR/3 (aquaculture OR fish OR shrimp)))) OR ("agroforestry farmer*" OR "agroforestry producer*" OR "tree crop farmer*" OR "fruit grower*" OR "forest farmer*" OR silvopastoral OR "silvo-pastoral" OR "tree-livestock farmer*" OR (((farmer* OR producer*) NEAR/3 (agroforestry OR "tree crop*" OR fruit OR forest OR silvopastoral OR "tree-livestock")))) OR (women OR female* OR "women farmer*" OR "female farmer*" OR youth OR "rural youth" OR "young farmer*" OR "young producer*" OR adolescent* OR "landless farmer*" OR "tenant farmer*" OR sharecropper* OR "contract farmer*" OR "Indigenous Peoples" OR "indigenous communit*" OR "tribal communit*" OR "ethnic minorit*" OR "agricultural laborer*" OR "farm worker*" OR "seasonal worker*" OR "migrant farmworker*" OR (((farmer* OR producer* OR household* OR laborer* OR worker*) NEAR/3 (women OR female* OR youth OR adolescent* OR landless OR tenant OR sharecropper* OR contract OR Indigenous OR tribal OR "ethnic minorit*" OR migrant OR seasonal)))))) AND (((adapt* NEAR/5 (practice* OR behav* OR decision* OR strateg* OR adjust* OR chang*) OR uptake OR adoption OR participat* OR "participatory process*" OR "adaptive management" OR learning* OR "learning process*" OR "feedback loop*" OR "feedback mechanism*" OR (governance NEAR/5 (adapt* OR climate)) OR extension* OR insurance*) OR ("adaptive capacity" OR (capacity NEAR/3 (build* OR strengthen* OR expand* OR improv*)) OR (knowledge NEAR/3 (adapt* OR climate OR resilience)) OR information* OR training*) OR ((resilien* NEAR/5 (outcome* OR capacity OR livelihood* OR improv* OR effect*)) OR (adapt* NEAR/5 (outcome* OR impact* OR effect* OR capacity OR livelihood*)) OR (reduc* NEAR/5 (risk OR "climate risk*" OR vulnerabilit* OR exposure*)) OR wellbeing OR "well-being") OR (livelihood* OR "coping strateg*" OR "coping capacit*" OR (income NEAR/5 (stability OR variability OR loss*)) OR (yield NEAR/5 (stability OR improve OR increase OR variability OR loss*)) OR increase*) OR (maladapt* OR "maladaptive outcome*" OR "maladaptive adaptation"))) AND ((((climate NEAR/3 (change OR adapt* OR resilien* OR risk OR hazard* OR variability OR exposure* OR shock*)) OR "climate-resilient*" OR "climate-smart agriculture" OR CSA) OR ((extreme* NEAR/3 (climate OR heat OR weather OR rainfall OR aridity)) OR heatwave* OR "heat wave*" OR "heat stress" OR frost OR "cold spell" OR "heavy rainfall" OR "intense rainfall" OR drought* OR "water scarcity" OR "water stress" OR "aridity events" OR "dry spell" OR flood* OR "storm surge" OR "coastal erosion" OR "salinity intrusion" OR "saline intrusion" OR cyclone* OR hurricane* OR typhoon* OR (wind NEAR/3 (storm OR gust* OR intens*)) OR "weather shock*" OR disaster*))) AND (((agricultur* OR farm* OR crop* OR "food production" OR "livestock rearing" OR "animal husbandry" OR rangeland* OR fishery OR fisheries OR aquacultur* OR forestry OR agroforestry OR "tree crop*" OR silvopastoral OR "silvo-pastoral" OR "crop-livestock system" OR "mixed crop-livestock" OR "mixed farming" OR agroecolog* OR "integrated farming system*" OR irrigation*))) AND (((afghanistan OR albania OR algeria OR "american samoa" OR angola OR "antigua and barbuda" OR antigua OR barbuda OR argentina OR armenia OR aruba OR azerbaijan OR bahrain OR bangladesh OR barbados OR "republic of belarus" OR belarus OR byelarus OR belorussia OR belize OR "british honduras" OR benin OR dahomey OR bhutan OR bolivia OR bosnia OR herzegovina OR botswana OR bechuanaland OR brazil OR brasil OR bulgaria OR "burkina faso" OR "burkina fasso" OR "upper volta" OR burundi OR urundi OR "cabo verde" OR "cape verde" OR cambodia OR kampuchea OR "khmer republic" OR cameroon OR cameron OR cameroun OR "central african republic" OR "ubangi shari" OR chad OR chile OR china OR colombia OR comoros OR "comoro islands" OR "iles comores" OR mayotte OR congo OR zaire OR "costa rica" OR "cote d'ivoire" OR "cote d ivoire" OR "cote divoire" OR "ivory coast" OR croatia OR cuba OR cyprus OR "czech republic" OR czechoslovakia OR djibouti OR "french somaliland" OR dominica OR "dominican republic" OR ecuador OR egypt OR "united arab republic" OR "el salvador" OR "equatorial guinea" OR "spanish guinea" OR eritrea OR estonia OR eswatini OR swaziland OR ethiopia OR fiji OR gabon OR "gabonese republic" OR gambia OR "georgia (republic)" OR ghana OR "gold coast" OR gibraltar OR greece OR grenada OR guam OR guatemala OR guinea OR "guinea bissau" OR guyana OR "british guiana" OR haiti OR hispaniola OR honduras OR hungary OR india OR indonesia OR timor OR iran OR iraq OR "isle of man" OR jamaica OR jordan OR kazakhstan OR kenya OR korea OR kosovo OR kyrgyzstan OR kirghizia OR kirgizstan OR "kyrgyz republic" OR laos OR "lao pdr" OR "lao people's democratic republic" OR latvia OR lebanon OR "lebanese republic" OR lesotho OR basutoland OR liberia OR libya OR "libyan arab jamahiriya" OR lithuania OR macau OR macao OR macedonia OR madagascar OR "malagasy republic" OR malawi OR nyasaland OR malaysia OR "malay federation" OR "malaya federation" OR maldives OR mali OR malta OR micronesia OR kiribati OR "marshall islands" OR nauru OR "northern mariana islands" OR palau OR tuvalu OR mauritania OR mauritius OR mexico OR moldova OR mongolia OR montenegro OR morocco OR ifni OR mozambique OR "portuguese east africa" OR myanmar OR burma OR namibia OR nepal OR "netherlands antilles" OR nicaragua OR niger OR nigeria OR oman OR muscat OR pakistan OR panama OR "new guinea" OR paraguay OR peru OR philippines OR philipines OR phillipines OR phillippines OR poland OR "polish people's republic" OR portugal OR "portuguese republic" OR "puerto rico" OR romania OR russia OR "russian federation" OR ussr OR "soviet union" OR "union of soviet socialist republics" OR rwanda OR ruanda OR samoa OR "pacific islands" OR polynesia OR "samoan islands" OR "navigator island" OR "navigator islands" OR "sao tome and principe" OR "saudi arabia" OR senegal OR serbia OR seychelles OR "sierra leone" OR slovakia OR "slovak republic" OR slovenia OR melanesia OR "solomon island" OR "solomon islands" OR "norfolk island" OR "norfolk islands" OR somalia OR "south africa" OR "sri lanka" OR ceylon OR "saint kitts and nevis" OR "st. kitts and nevis" OR "saint lucia" OR "st. lucia" OR "saint vincent and the grenadines" OR "saint vincent" OR "st. vincent" OR grenadines OR sudan OR suriname OR surinam OR "dutch guiana" OR "netherlands guiana" OR syria OR "syrian arab republic" OR tajikistan OR tadjikistan OR tadzhikistan OR tanzania OR tanganyika OR thailand OR siam OR "timor leste" OR "east timor" OR togo OR "togolese republic" OR tonga OR trinidad OR tobago OR tunisia OR turkey OR turkmenistan OR uganda OR ukraine OR uruguay OR uzbekistan OR vanuatu OR "new hebrides" OR venezuela OR vietnam OR "viet nam" OR "west bank" OR gaza OR palestine OR yemen OR yugoslavia OR zambia OR zimbabwe OR "northern rhodesia") OR (africa OR "sub-saharan africa" OR "north africa" OR "west africa" OR "east africa" OR "southern africa" OR "central africa" OR sahara OR magreb OR maghrib OR "latin america" OR "central america" OR "south america" OR "south and central america" OR caribbean OR "west indies" OR "middle east" OR mena OR asia OR "central asia" OR "east asia" OR "south asia" OR "southeast asia" OR "south east asia" OR "south eastern asia" OR "western asia" OR "north asia" OR "northern asia" OR "eastern europe" OR "east europe" OR "europe, eastern" OR "indian ocean" OR "indian ocean islands" OR "pacific islands" OR "pacific islander*" OR "global south") OR (lmic OR lmics OR "developing country" OR "developing countries" OR "developing nation*" OR "developing population*" OR "developing world" OR "less developed countr*" OR "less developed nation*" OR "less developed population*" OR "less developed world" OR "lesser developed countr*" OR "lesser developed nation*" OR "lesser developed population*" OR "lesser developed world" OR "under developed countr*" OR "under developed nation*" OR "under developed population*" OR "under developed world" OR "underdeveloped countr*" OR "underdeveloped nation*" OR "underdeveloped population*" OR "underdeveloped world" OR "low income countr*" OR "low income nation*" OR "low income population*" OR "lower income countr*" OR "lower income nation*" OR "lower income population*" OR "middle income countr*" OR "middle income nation*" OR "middle income population*" OR "poor countr*" OR "poor nation*" OR "poor population*" OR "third world" OR "emerging econom*" OR "emerging economies" OR "emerging nation*"))) AND (((indicator* OR metric* OR "measurement framework*" OR monitor* OR evaluat* OR assess* OR "impact evaluation" OR "results-based management" OR "results based management" OR "M&E" OR MEL OR effectiv* OR impact* OR (measur* NEAR/5 (adapt* OR resilience OR outcome*)) OR index OR indices OR "data collection" OR survey* OR (participatory NEAR/3 (monitoring OR method* OR assessment)) OR "mixed methods" OR regression*))))
```

**If the query is rejected for length**, use the 6-row split version in `scripts/data/multidatabase/wos/search_string_wos_split.txt` — paste each `TS=(...)` row into a separate Advanced Search field, all joined with AND.

**Steps:**
1. Go to https://www.webofscience.com — authenticates automatically on Cornell campus/VPN
2. Click **Advanced Search**, paste the full query above
3. Apply date filter: **Publication Year 2005–present**
4. Note the total hit count in the table above
5. Export in batches of 500: **Export** → **Other File Formats** → Record content: Full Record → File format: **RIS format** → Records: 1–500, then 501–1000, etc.
6. Upload to Drive: `multidatabase/wos/wos_export_1.ris`, `wos_export_2.ris`, ...

---

## CAB Abstracts (Cornell campus or VPN)

Full string file: `scripts/data/multidatabase/cab/search_string_ebsco.txt`

Adapted from pipeline query: `TITLE-ABS-KEY` → `TX`, `W/N` → `N/N` (EBSCOhost proximity syntax).

**Query:**

```
TX=((((smallholder* OR "small holder*" OR "small-scale farmer*" OR "small-scale producer*" OR "subsistence farmer*" OR "farmers in low-income countr*" OR "small-scale farmers in low-income countr*" OR "marginal farmer*" OR "resource-poor farmer*" OR  ((farmer* OR producer*) N3 (smallholder* OR "small-scale" OR subsistence OR marginal* OR "resource-poor"))) OR ("rural household*" OR "farm household*" OR "agricultural household*" OR "family farmer*" OR "low-income household*" OR "poor household*" OR "female-headed household*" OR "women-headed household*" OR ((household* OR family OR farmer*) N3 (rural OR agricultural OR farm* OR "low-income" OR poor OR female-headed OR women-headed))) OR (pastoralist* OR "agro-pastoralist*" OR "agro pastoralist*" OR herder* OR "livestock keeper*" OR "livestock farmer*" OR "livestock producer*" OR "dairy farmer*" OR "crop-livestock farmer*" OR (pastoralist* OR herder* OR ((farmer* OR producer* OR keeper*) N3 (livestock OR dairy OR cattle OR crop-livestock)) OR "agro-pastoral*")) OR (fisher* OR "small-scale fisher*" OR "small scale fisher*" OR fisherfolk OR "artisanal fisher*" OR "aquaculture producer*" OR "fish farmer*" OR "shrimp farmer*" OR ((fisher* N3 (small-scale OR smallscale OR artisanal)) OR ((farmer* OR producer*) N3 (aquaculture OR fish OR shrimp)))) OR ("agroforestry farmer*" OR "agroforestry producer*" OR "tree crop farmer*" OR "fruit grower*" OR "forest farmer*" OR silvopastoral OR "silvo-pastoral" OR "tree-livestock farmer*" OR (((farmer* OR producer*) N3 (agroforestry OR "tree crop*" OR fruit OR forest OR silvopastoral OR "tree-livestock")))) OR (women OR female* OR "women farmer*" OR "female farmer*" OR youth OR "rural youth" OR "young farmer*" OR "young producer*" OR adolescent* OR "landless farmer*" OR "tenant farmer*" OR sharecropper* OR "contract farmer*" OR "Indigenous Peoples" OR "indigenous communit*" OR "tribal communit*" OR "ethnic minorit*" OR "agricultural laborer*" OR "farm worker*" OR "seasonal worker*" OR "migrant farmworker*" OR (((farmer* OR producer* OR household* OR laborer* OR worker*) N3 (women OR female* OR youth OR adolescent* OR landless OR tenant OR sharecropper* OR contract OR Indigenous OR tribal OR "ethnic minorit*" OR migrant OR seasonal)))))) AND (((adapt* N5 (practice* OR behav* OR decision* OR strateg* OR adjust* OR chang*) OR uptake OR adoption OR participat* OR "participatory process*" OR "adaptive management" OR learning* OR "learning process*" OR "feedback loop*" OR "feedback mechanism*" OR (governance N5 (adapt* OR climate)) OR extension* OR insurance*) OR ("adaptive capacity" OR (capacity N3 (build* OR strengthen* OR expand* OR improv*)) OR (knowledge N3 (adapt* OR climate OR resilience)) OR information* OR training*) OR ((resilien* N5 (outcome* OR capacity OR livelihood* OR improv* OR effect*)) OR (adapt* N5 (outcome* OR impact* OR effect* OR capacity OR livelihood*)) OR (reduc* N5 (risk OR "climate risk*" OR vulnerabilit* OR exposure*)) OR wellbeing OR "well-being") OR (livelihood* OR "coping strateg*" OR "coping capacit*" OR (income N5 (stability OR variability OR loss*)) OR (yield N5 (stability OR improve OR increase OR variability OR loss*)) OR increase*) OR (maladapt* OR "maladaptive outcome*" OR "maladaptive adaptation"))) AND ((((climate N3 (change OR adapt* OR resilien* OR risk OR hazard* OR variability OR exposure* OR shock*)) OR "climate-resilient*" OR "climate-smart agriculture" OR CSA) OR ((extreme* N3 (climate OR heat OR weather OR rainfall OR aridity)) OR heatwave* OR "heat wave*" OR "heat stress" OR frost OR "cold spell" OR "heavy rainfall" OR "intense rainfall" OR drought* OR "water scarcity" OR "water stress" OR "aridity events" OR "dry spell" OR flood* OR "storm surge" OR "coastal erosion" OR "salinity intrusion" OR "saline intrusion" OR cyclone* OR hurricane* OR typhoon* OR (wind N3 (storm OR gust* OR intens*)) OR "weather shock*" OR disaster*))) AND (((agricultur* OR farm* OR crop* OR "food production" OR "livestock rearing" OR "animal husbandry" OR rangeland* OR fishery OR fisheries OR aquacultur* OR forestry OR agroforestry OR "tree crop*" OR silvopastoral OR "silvo-pastoral" OR "crop-livestock system" OR "mixed crop-livestock" OR "mixed farming" OR agroecolog* OR "integrated farming system*" OR irrigation*))) AND (((afghanistan OR albania OR algeria OR "american samoa" OR angola OR "antigua and barbuda" OR antigua OR barbuda OR argentina OR armenia OR aruba OR azerbaijan OR bahrain OR bangladesh OR barbados OR "republic of belarus" OR belarus OR byelarus OR belorussia OR belize OR "british honduras" OR benin OR dahomey OR bhutan OR bolivia OR bosnia OR herzegovina OR botswana OR bechuanaland OR brazil OR brasil OR bulgaria OR "burkina faso" OR "burkina fasso" OR "upper volta" OR burundi OR urundi OR "cabo verde" OR "cape verde" OR cambodia OR kampuchea OR "khmer republic" OR cameroon OR cameron OR cameroun OR "central african republic" OR "ubangi shari" OR chad OR chile OR china OR colombia OR comoros OR "comoro islands" OR "iles comores" OR mayotte OR congo OR zaire OR "costa rica" OR "cote d'ivoire" OR "cote d ivoire" OR "cote divoire" OR "ivory coast" OR croatia OR cuba OR cyprus OR "czech republic" OR czechoslovakia OR djibouti OR "french somaliland" OR dominica OR "dominican republic" OR ecuador OR egypt OR "united arab republic" OR "el salvador" OR "equatorial guinea" OR "spanish guinea" OR eritrea OR estonia OR eswatini OR swaziland OR ethiopia OR fiji OR gabon OR "gabonese republic" OR gambia OR "georgia (republic)" OR ghana OR "gold coast" OR gibraltar OR greece OR grenada OR guam OR guatemala OR guinea OR "guinea bissau" OR guyana OR "british guiana" OR haiti OR hispaniola OR honduras OR hungary OR india OR indonesia OR timor OR iran OR iraq OR "isle of man" OR jamaica OR jordan OR kazakhstan OR kenya OR korea OR kosovo OR kyrgyzstan OR kirghizia OR kirgizstan OR "kyrgyz republic" OR laos OR "lao pdr" OR "lao people's democratic republic" OR latvia OR lebanon OR "lebanese republic" OR lesotho OR basutoland OR liberia OR libya OR "libyan arab jamahiriya" OR lithuania OR macau OR macao OR macedonia OR madagascar OR "malagasy republic" OR malawi OR nyasaland OR malaysia OR "malay federation" OR "malaya federation" OR maldives OR mali OR malta OR micronesia OR kiribati OR "marshall islands" OR nauru OR "northern mariana islands" OR palau OR tuvalu OR mauritania OR mauritius OR mexico OR moldova OR mongolia OR montenegro OR morocco OR ifni OR mozambique OR "portuguese east africa" OR myanmar OR burma OR namibia OR nepal OR "netherlands antilles" OR nicaragua OR niger OR nigeria OR oman OR muscat OR pakistan OR panama OR "new guinea" OR paraguay OR peru OR philippines OR philipines OR phillipines OR phillippines OR poland OR "polish people's republic" OR portugal OR "portuguese republic" OR "puerto rico" OR romania OR russia OR "russian federation" OR ussr OR "soviet union" OR "union of soviet socialist republics" OR rwanda OR ruanda OR samoa OR "pacific islands" OR polynesia OR "samoan islands" OR "navigator island" OR "navigator islands" OR "sao tome and principe" OR "saudi arabia" OR senegal OR serbia OR seychelles OR "sierra leone" OR slovakia OR "slovak republic" OR slovenia OR melanesia OR "solomon island" OR "solomon islands" OR "norfolk island" OR "norfolk islands" OR somalia OR "south africa" OR "sri lanka" OR ceylon OR "saint kitts and nevis" OR "st. kitts and nevis" OR "saint lucia" OR "st. lucia" OR "saint vincent and the grenadines" OR "saint vincent" OR "st. vincent" OR grenadines OR sudan OR suriname OR surinam OR "dutch guiana" OR "netherlands guiana" OR syria OR "syrian arab republic" OR tajikistan OR tadjikistan OR tadzhikistan OR tanzania OR tanganyika OR thailand OR siam OR "timor leste" OR "east timor" OR togo OR "togolese republic" OR tonga OR trinidad OR tobago OR tunisia OR turkey OR turkmenistan OR uganda OR ukraine OR uruguay OR uzbekistan OR vanuatu OR "new hebrides" OR venezuela OR vietnam OR "viet nam" OR "west bank" OR gaza OR palestine OR yemen OR yugoslavia OR zambia OR zimbabwe OR "northern rhodesia") OR (africa OR "sub-saharan africa" OR "north africa" OR "west africa" OR "east africa" OR "southern africa" OR "central africa" OR sahara OR magreb OR maghrib OR "latin america" OR "central america" OR "south america" OR "south and central america" OR caribbean OR "west indies" OR "middle east" OR mena OR asia OR "central asia" OR "east asia" OR "south asia" OR "southeast asia" OR "south east asia" OR "south eastern asia" OR "western asia" OR "north asia" OR "northern asia" OR "eastern europe" OR "east europe" OR "europe, eastern" OR "indian ocean" OR "indian ocean islands" OR "pacific islands" OR "pacific islander*" OR "global south") OR (lmic OR lmics OR "developing country" OR "developing countries" OR "developing nation*" OR "developing population*" OR "developing world" OR "less developed countr*" OR "less developed nation*" OR "less developed population*" OR "less developed world" OR "lesser developed countr*" OR "lesser developed nation*" OR "lesser developed population*" OR "lesser developed world" OR "under developed countr*" OR "under developed nation*" OR "under developed population*" OR "under developed world" OR "underdeveloped countr*" OR "underdeveloped nation*" OR "underdeveloped population*" OR "underdeveloped world" OR "low income countr*" OR "low income nation*" OR "low income population*" OR "lower income countr*" OR "lower income nation*" OR "lower income population*" OR "middle income countr*" OR "middle income nation*" OR "middle income population*" OR "poor countr*" OR "poor nation*" OR "poor population*" OR "third world" OR "emerging econom*" OR "emerging economies" OR "emerging nation*"))) AND (((indicator* OR metric* OR "measurement framework*" OR monitor* OR evaluat* OR assess* OR "impact evaluation" OR "results-based management" OR "results based management" OR "M&E" OR MEL OR effectiv* OR impact* OR (measur* N5 (adapt* OR resilience OR outcome*)) OR index OR indices OR "data collection" OR survey* OR (participatory N3 (monitoring OR method* OR assessment)) OR "mixed methods" OR regression*))))
```

**Steps:**
1. Go to https://www.cabdirect.org — log in via Cornell institutional access
2. Click **Advanced Search**, paste the full query above
3. Apply date filter: **2005–present**
4. Note the total hit count in the table above
5. Export in batches of 200: **Download** → **RIS** (select all on page → export → next page → repeat)
6. Upload to Drive: `multidatabase/cab/cab_export_1.ris`, `cab_export_2.ris`, ...

---

## Academic Search Premier (Cornell campus or VPN)

Full string file: `scripts/data/multidatabase/asp/search_string_ebsco.txt`

**Uses the identical query as CAB Abstracts** — EBSCOhost syntax is the same across databases.

**Steps:**
1. Go to https://search.ebscohost.com — select **Academic Search Premier** from the database list
2. Click **Advanced Search**, paste the same `TX=(...)` query from CAB above
3. Apply date filter: **2005–present**
4. Note the total hit count in the table above
5. Export in batches of 200: **Export** → **Direct Export in RIS format** (select all on page → repeat)
6. Upload to Drive: `multidatabase/asp/asp_export_1.ris`, `asp_export_2.ris`, ...
