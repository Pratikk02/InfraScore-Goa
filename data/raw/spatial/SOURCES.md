# Spatial data sources

| File | Content | Source | Licence |
|---|---|---|---|
| `geonames_goa_cities500.csv` | Goa places with population > 500 (name, lat, lon, population) | GeoNames `cities500`, as bundled in the `geonamescache` 3.0.2 package on PyPI | CC BY 4.0 (GeoNames) |
| `town_coordinates.csv` | Coordinates for the 69 analysed towns (+ Orgao, shown on the map only), matched by name; 53 of 69 matches also agree exactly on the 2011 population | derived from the file above | CC BY 4.0 |
| `goa_districts_datta07.geojson` | Goa district polygons (used only as a state outline on the map) | github.com/datta07/INDIAN-SHAPEFILES (`STATES/GOA/GOA_DISTRICTS.geojson`) | curated public data, see repository |
| `contemporary_facilities.csv` | Post-2011 public hospitals / health centres and fire stations, with the place they are located in and the source of each record | Goa Legislative Assembly reply (Directorate of Health Services bed-capacity list) and Directorate of Fire & Emergency Services publications | public records |

Omitted or approximate records: the 15-bed Primary Health Centre at Dharbandora is omitted (no verifiable coordinates); the T.B. Hospital (specialty) is excluded; Bali PHC (Balli railway station), Cansaulim PHC (adjacent village Cuelim) and Kundaim fire station are placed at approximate coordinates, as recorded in `contemporary_facilities.csv`.

The polygon files place Ponda taluka in South Goa, whereas the Census 2011 Town Directory lists it under North Goa, so district polygons are drawn without labels and the census district is used throughout the analysis.
