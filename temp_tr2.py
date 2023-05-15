import csv
import simplekml

with open("./tr2_two_stage_trimmed.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    coords = []
    last_timestamp = 0
    for row in reader:
        latitude = float(row['latitude1'])
        longitude = float(row['longitude1'])
        altitude = float(row['altitude_estimate']) - float(reader.fieldnames.index("altitude_estimate"))
        altitude -= 45
        timestamp = float(row['timestamp'])
        #print(altitude-45)
        print(latitude,longitude)
        if timestamp - last_timestamp < 10:
            continue

        coords.append((longitude, latitude, altitude))

        last_timestamp = timestamp

kml = simplekml.Kml()
ls = kml.newlinestring(name="MRAS GPS track")
ls.coords = coords
ls.altitudemode = simplekml.AltitudeMode.relativetoground
ls.style.linestyle.color = "ff3c14dc"
ls.style.linestyle.width = 3
ls.polystyle.outline = 0
ls.polystyle.color = "643c14dc"
ls.extrude = 1
kml.save('tr2_output.kml')
