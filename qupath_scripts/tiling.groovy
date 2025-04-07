import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles_progetto', name)
mkdirs(pathOutput)

double requestedPixelSize = 1
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Set the background label to 0 (black)
    .downsample(downsample)    
    .addLabel('Tissue', 1, ColorTools.RED) 
    .addLabel('BileDuct', 2, ColorTools.WHITE)      
    .multichannelOutput(false)                         
    .build()
new TileExporter(imageData)
    .downsample(downsample)     
    .imageExtension('.png')     
    .tileSize(256)              
    .labeledServer(labelServer) 
    .annotatedTilesOnly(true)   
    .overlap(0)                 
    .writeTiles(pathOutput)     

print 'Done!'