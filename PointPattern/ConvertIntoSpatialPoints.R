# install.packages('maptools')

require(maptools)
require(spatstat)

# Bramblecanes is a dta set in ppp format from spatstat
data("bramblecanes")

# Convert the data to SpatialPoints, and plot them
bc.spformat <-as(bramblecanes, 'SpatialPoints')
plot(bc.spformat)

# It is also possible to extract the study polygon
# referred to as a window in spatstat terminology
# Here it is just a rectangle ..

bc.win <- as(bramblecanes$window, 'SpatialPolygons')
plot(bc.win, add=TRUE)