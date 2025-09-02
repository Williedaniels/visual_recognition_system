import { useState, useCallback } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  Upload, 
  Eye, 
  Users, 
  Search, 
  BarChart3, 
  Image as ImageIcon,
  Camera,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react'
import './App.css'

const API_BASE_URL = 'http://localhost:5000/api'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState({})
  const [activeTab, setActiveTab] = useState('classify')
  const [error, setError] = useState(null)

  // Convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result)
      reader.onerror = error => reject(error)
    })
  }

  // Handle image upload
  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
      setImagePreview(URL.createObjectURL(file))
      setResults({})
      setError(null)
    }
  }, [])

  // API call helper
  const callAPI = async (endpoint, imageData) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
      })
      
      if (!response.ok) {
        throw new Error(`API call failed: ${response.statusText}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('API Error:', error)
      throw error
    }
  }

  // Image classification
  const classifyImage = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    setError(null)
    
    try {
      const base64Image = await fileToBase64(selectedImage)
      const result = await callAPI('/classify', base64Image)
      setResults(prev => ({ ...prev, classification: result }))
    } catch (error) {
      setError(`Classification failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Object detection
  const detectObjects = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    setError(null)
    
    try {
      const base64Image = await fileToBase64(selectedImage)
      const result = await callAPI('/detect_objects', base64Image)
      setResults(prev => ({ ...prev, objects: result }))
    } catch (error) {
      setError(`Object detection failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Face recognition
  const recognizeFaces = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    setError(null)
    
    try {
      const base64Image = await fileToBase64(selectedImage)
      const result = await callAPI('/recognize_faces', base64Image)
      setResults(prev => ({ ...prev, faces: result }))
    } catch (error) {
      setError(`Face recognition failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Comprehensive analysis
  const analyzeImage = async () => {
    if (!selectedImage) return
    
    setLoading(true)
    setError(null)
    
    try {
      const base64Image = await fileToBase64(selectedImage)
      const result = await callAPI('/analyze_image', base64Image)
      setResults(prev => ({ ...prev, analysis: result }))
    } catch (error) {
      setError(`Image analysis failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Render classification results
  const renderClassificationResults = () => {
    const data = results.classification
    if (!data) return null

    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Classification Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-medium">Prediction:</span>
                <Badge variant="secondary">{data.prediction}</Badge>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Confidence:</span>
                  <span>{(data.confidence * 100).toFixed(1)}%</span>
                </div>
                <Progress value={data.confidence * 100} className="h-2" />
              </div>
              
              {data.scene_analysis && (
                <div className="mt-4 p-3 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2">Scene Analysis</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>Entropy: {data.scene_analysis.complexity?.entropy?.toFixed(2)}</div>
                    <div>Contrast: {data.scene_analysis.complexity?.contrast}</div>
                    <div>Quality: {data.scene_analysis.quality?.grade}</div>
                    <div>Score: {(data.scene_analysis.quality?.score * 100).toFixed(0)}%</div>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Render object detection results
  const renderObjectResults = () => {
    const data = results.objects
    if (!data) return null

    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Object Detection Results
            </CardTitle>
            <CardDescription>
              Found {data.num_detections} object(s)
            </CardDescription>
          </CardHeader>
          <CardContent>
            {data.result_image && (
              <div className="mb-4">
                <img 
                  src={data.result_image} 
                  alt="Detection results" 
                  className="w-full max-w-md mx-auto rounded-lg border"
                />
              </div>
            )}
            
            {data.detections && data.detections.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium">Detected Objects:</h4>
                {data.detections.map((detection, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                    <span className="font-medium">{detection.class_name}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">
                        {(detection.confidence * 100).toFixed(1)}%
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        {detection.width}×{detection.height}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    )
  }

  // Render face recognition results
  const renderFaceResults = () => {
    const data = results.faces
    if (!data) return null

    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Face Recognition Results
            </CardTitle>
            <CardDescription>
              Found {data.num_faces} face(s)
            </CardDescription>
          </CardHeader>
          <CardContent>
            {data.result_image && (
              <div className="mb-4">
                <img 
                  src={data.result_image} 
                  alt="Face recognition results" 
                  className="w-full max-w-md mx-auto rounded-lg border"
                />
              </div>
            )}
            
            {data.faces && data.faces.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium">Recognized Faces:</h4>
                {data.faces.map((face, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                    <span className="font-medium">{face.name}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">
                        {(face.confidence * 100).toFixed(1)}%
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        {face.bbox.width}×{face.bbox.height}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    )
  }

  // Render comprehensive analysis results
  const renderAnalysisResults = () => {
    const data = results.analysis
    if (!data) return null

    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Comprehensive Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Quality Assessment */}
              <div className="space-y-3">
                <h4 className="font-medium">Quality Assessment</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Overall Quality:</span>
                    <Badge variant={data.quality_assessment?.overall_grade === 'excellent' ? 'default' : 'secondary'}>
                      {data.quality_assessment?.overall_grade}
                    </Badge>
                  </div>
                  <Progress 
                    value={data.quality_assessment?.overall_score * 100} 
                    className="h-2" 
                  />
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex items-center gap-1">
                      {data.quality_assessment?.blur_analysis?.is_blurry ? 
                        <AlertCircle className="h-3 w-3 text-orange-500" /> : 
                        <CheckCircle className="h-3 w-3 text-green-500" />
                      }
                      Blur: {data.quality_assessment?.blur_analysis?.blur_level}
                    </div>
                    <div className="flex items-center gap-1">
                      {data.quality_assessment?.contrast_analysis?.is_low_contrast ? 
                        <AlertCircle className="h-3 w-3 text-orange-500" /> : 
                        <CheckCircle className="h-3 w-3 text-green-500" />
                      }
                      Contrast: {data.quality_assessment?.contrast_analysis?.contrast_level}
                    </div>
                  </div>
                </div>
              </div>

              {/* Scene Composition */}
              <div className="space-y-3">
                <h4 className="font-medium">Scene Composition</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Dimensions:</span>
                    <span>{data.scene_composition?.dimensions?.width}×{data.scene_composition?.dimensions?.height}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Aspect Ratio:</span>
                    <span>{data.scene_composition?.dimensions?.aspect_ratio?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Complexity:</span>
                    <span>{data.scene_composition?.complexity?.entropy?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Edge Density:</span>
                    <span>{(data.scene_composition?.texture_analysis?.edge_density * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Content Detection */}
              <div className="space-y-3">
                <h4 className="font-medium">Content Detection</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Faces Detected:</span>
                    <Badge variant="outline">{data.content_detection?.num_faces || 0}</Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    {data.content_detection?.has_faces ? 
                      <CheckCircle className="h-4 w-4 text-green-500" /> : 
                      <AlertCircle className="h-4 w-4 text-gray-400" />
                    }
                    <span className="text-sm">Contains faces</span>
                  </div>
                </div>
              </div>

              {/* Feature Summary */}
              <div className="space-y-3">
                <h4 className="font-medium">Feature Summary</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Color Diversity:</span>
                    <span>{data.feature_summary?.color_diversity?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Texture Complexity:</span>
                    <span>{data.feature_summary?.texture_complexity?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Shape Regularity:</span>
                    <span>{data.feature_summary?.shape_regularity?.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-lg">
              <Eye className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Visual Recognition System</h1>
              <p className="text-muted-foreground">
                Advanced computer vision and machine learning for image analysis
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Image
                </CardTitle>
                <CardDescription>
                  Select an image to analyze with our AI models
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      id="image-upload"
                    />
                    <label htmlFor="image-upload" className="cursor-pointer">
                      <div className="space-y-2">
                        <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground" />
                        <div className="text-sm text-muted-foreground">
                          Click to upload or drag and drop
                        </div>
                      </div>
                    </label>
                  </div>

                  {imagePreview && (
                    <div className="space-y-3">
                      <img 
                        src={imagePreview} 
                        alt="Preview" 
                        className="w-full rounded-lg border"
                      />
                      
                      <div className="grid grid-cols-2 gap-2">
                        <Button 
                          onClick={classifyImage} 
                          disabled={loading}
                          variant="outline"
                          size="sm"
                        >
                          {loading && activeTab === 'classify' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Eye className="h-4 w-4" />
                          )}
                          Classify
                        </Button>
                        
                        <Button 
                          onClick={detectObjects} 
                          disabled={loading}
                          variant="outline"
                          size="sm"
                        >
                          {loading && activeTab === 'objects' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Search className="h-4 w-4" />
                          )}
                          Detect
                        </Button>
                        
                        <Button 
                          onClick={recognizeFaces} 
                          disabled={loading}
                          variant="outline"
                          size="sm"
                        >
                          {loading && activeTab === 'faces' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Users className="h-4 w-4" />
                          )}
                          Faces
                        </Button>
                        
                        <Button 
                          onClick={analyzeImage} 
                          disabled={loading}
                          variant="outline"
                          size="sm"
                        >
                          {loading && activeTab === 'analysis' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <BarChart3 className="h-4 w-4" />
                          )}
                          Analyze
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Features Overview */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Features
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-sm">
                    <Eye className="h-4 w-4 text-blue-500" />
                    <span>Image Classification</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <Search className="h-4 w-4 text-green-500" />
                    <span>Object Detection</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <Users className="h-4 w-4 text-purple-500" />
                    <span>Face Recognition</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <BarChart3 className="h-4 w-4 text-orange-500" />
                    <span>Quality Analysis</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <Camera className="h-4 w-4 text-red-500" />
                    <span>Scene Composition</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {error && (
              <Alert className="mb-6" variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="classify">Classification</TabsTrigger>
                <TabsTrigger value="objects">Objects</TabsTrigger>
                <TabsTrigger value="faces">Faces</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
              </TabsList>

              <TabsContent value="classify" className="mt-6">
                {results.classification ? (
                  renderClassificationResults()
                ) : (
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center text-muted-foreground">
                        Upload an image and click "Classify" to see classification results
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="objects" className="mt-6">
                {results.objects ? (
                  renderObjectResults()
                ) : (
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center text-muted-foreground">
                        Upload an image and click "Detect" to see object detection results
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="faces" className="mt-6">
                {results.faces ? (
                  renderFaceResults()
                ) : (
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center text-muted-foreground">
                        Upload an image and click "Faces" to see face recognition results
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="analysis" className="mt-6">
                {results.analysis ? (
                  renderAnalysisResults()
                ) : (
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-center text-muted-foreground">
                        Upload an image and click "Analyze" to see comprehensive analysis
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

