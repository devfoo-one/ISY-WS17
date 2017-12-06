# Fragen zu Abgabe 3.4.

## Was passiert, wenn Sie das Distanz-Filtering aus dem gegebenen Sourcecode nicht nutzen?

Wenn die Matches nicht mehr nach ihrer Distanz gefiltert werden, werden in schwierigen Situationen vermehrt "entfernt ähnliche" statt gar keine Matches an die weiteren Codeteile durchgereicht. Das führt zu Fehlern bei der Erzeugung der Homography-Matrix, da diese nun "krampfhaft" versucht eine Transformation abzubilden die so gar nicht existiert.
Die Zeile `match[0].distance < match[1].distance * 0.75` tut dies, indem sie sicherstellt das sich die Entfernung des ersten nearest neighbour signifikant von der Entfernung zum zweiten unterscheidet.

## Wie sieht die OpenCV Datenstruktur der Matches aus?

Der Aufruf von `matches = bf.knnMatch(descriptorsFrame, descriptorsMarker, k=1)` erzeugt eine Liste von Matches. Diese enthält pro Descriptor des ersten Parameters (`descriptorsFrame`) ("Query-Seite") eine Liste mit `k` nearest neighbours aus `descriptorsMarker`.
Im Match befindet sich dann der Index des Query-Keypoints aus `descriptorFrame`, der Index des jeweils gematchten “Train”-Keypoints aus `descriptorsMarker` sowie die Distanz der beiden Vektoren.

## Was passiert wenn Sie den Thresholdwert in cv2.findHomography ändern und warum?

Der Threshold gibt den maximal zulässigen Projektionsfehler (in Pixeln) beim Finden der Homographiematrix zur Durchführung der perspektivischen Transformation an. Da es möglich wäre dass Features zwar inhaltlich nah beieinanderliegen (knn search), jedoch nicht durch eine perspektivische Transformation aufeinander abgebildet werden können (z.B. doppelte Features im Bild, Wölbung des Markerbildes, ...) ist dieser Wert notwendig.
Eine Erhöhung des Wertes sorgt für eine wählerischere Auswahl der aufeinander abzubildenen Features und damit zu stabileren Abbildungen.
Eine Veringerung sorgt für häufige "Glitches", d.h. merkwürdige Abbildungen welche nicht der Realität entsprechen.
