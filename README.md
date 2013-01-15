# Rayleigh: search images by multiple colors

Rayleigh is an open-source system for quickly searching large image collections by multiple colors, given as a palette or derived from a query image.
The system, presented as a [running website](#TODO) has three parts.

- The back end is in Python with numpy/scipy, and a library called FLANN.
It processes image URLs to extract color information to store in a searchable database.
- The web server is in Python, using the Flask package.
It provides REST access to the back end.
- The client is HTML/Javascript with some jQuery.
It provides the UI for selecting a color palette to search with, or an image, and requests data from the web server.

## Introduction

To me, the point of color is two-fold: the fine differences in hue and shading, and the interaction of multiples.
I wanted to be able to search large image collections not only by the presence of a single color, but by a composition of multiple colors---and to treat colors as more nuanced than mere hues.

### Prior Art

The Content-Based Image Retrieval (CBIR) field has done much work on image similarity in the past fifteen years.
Although I have not done extensive literature review, my impression of much of the field is that color similarity was seen as proxy for general visual similarity, not an end to itself, in the early work, and the later work has focused more on closing the "semantic gap" in terms of nouns and verbs, not in terms of color perception.

The [Idee Labs Multicolr Search](), a rather expensive commercial service, is the only implemented prior art that I found.

## Perception of Color

All visual perception is highly complex.
As countless optical illusions show, we perceive objects of the same size in the image as [different sizes](), see checkerboard squares of the same color in the image as [different colors](), and see lines where there [aren't any]().

In the last example, the Kanisza triangle, we see the "illusory contour" of lines connecting the actual line segments.
To get a machine to "see" these illusory lines is practically an unsolved problem, and so if we were developing a system to search images by shape, we would struggle mightily to return this as a result for 'triangle.'

So it is with color.
As artists have [long understood](#interaction_of_color), how we see a color depends on what other colors are around it.
Furthermore, we may perceive an object as being the same color in two images in which the actual manifested colors, in the pixel values on the screen or the pigments laid down by a brush, are totally different.
Think of your face: you perceive as being roughly the same color in a lightbulb-lit bathroom mirror, a photograph taken outside in bright sunlight, in a fluorescent office, and in the darkness of evening---yet the wavelengths of light as registered by your retina are quite different in all these situations.

And if that weren't enough, there is a layer of language on top of all this mess.
Different languages deliniate colors in slightly different ways (for example, Russian has two distinct "blues"), and it has been shown that it actually [affects perception of color difference]().

## Representing Color

The problems above are hard, but we have a simpler situation.
We don't have to deal with "color constancy" of objects, or with names of colors.
We simply want to see images of exactly the same color as an example image, or as a selected color.
Further, we won't even allow just any color to be selected; we will present a palette, or "codebook," of allowed colors from which the user may select their own query palette.

### Human Eye

In our eyes, there are two types of photorceptive cells: rods and cones.
Rods respond only to intensity of light, not its color, and are far more numerous.
Cones have three distinct types, each responding strongest to a specific wavelength of light.
Our perception of color is derived from the response rates of the three types of cones.

### RGB Color Space

On your computer, color is represented as three values: one for intensity of red, one for intensity of green, and one for intensity of blue.
The pixels in the display are composed of these three basic lights.
When all are fully on, the color is white; when all are off, the color is black.
All of the millions of colors that a modern computer is able to display come from mixing the three intensities.

The RGB system can be thought of as describing a three dimensional space, with the Red, Green, and Blue dimensions.
A point in that space, given by the three coordinates, is a color.
We can begin to think of distances between colors in this way, as a distance in 3D space between two points.

### HSV Color Space

An additive mixture of three primaries does not match our intutive model of color.
We can't easily visualize the effect of adding red, green, or blue to a color.
Additionally, the distances in RGB spaces do not match up to perceptual judgements.
That is, a color may be quite far away from another one in terms of RGB coordinates, but humans will reliably judge the two colors as quite similar; or the other way: two colors may *look* very different but be close together in RGB space.

The rainbow is what we usually visualize when we think of color: hues of the visible spectrum from the almost-infrared to the almost-ultraviolet, roughly divided into less than a dozen words.
A given hue can be imagined as more vibrant than baseline---deep red, midnight blue, gold---or as more pastel-like---pink lipstick, robin's egg blue, sun-burned grass.

The Hue-Saturation-Value color space is informed by this mental model, and strives to have one dimension corresponding to our intuitive notion of hue, and two dimensions which set the vibrancy or lightness.
A point in HSV space is therefore more easily interpretable than a point in RGB space.

### Perceptually Uniform Color Spaces

Although it better matches our mental model of color, HSV space still suffers from the same misalignment of perceptual judgements to 3D distance of colors as RGB space.

The international standards agency has set out an alternative color space, with the explicit goal of making distances in the color space correspond to human judgments of color similarity.
Actually, it couldn't decide between two: CIELab and CIELuv.

CIELab, roughly, is formed by an oval with two axes: a and b, which correspond to the "opponent" colors of Yellow-Blue and Red-Green.
The opponent colors are so named because of the [opponent process](http://en.wikipedia.org/wiki/Opponent_process) theory, which posits that color perception comes from the *difference* in activation rates of the three types of cones in the retina.
The opponent colors have no in-between point: we can imagine a point between blue and red, but not between blue and yellow; between red and yellow, but not between red and green.
The third dimension of Lab space is lightness, which is approximately self-describing.

In the Lab space, simple Euclidean distance between two colors (corresponding to the intuitive notion of a distance between 3D points) is a good approximation to perceptual judgements of their difference.

## Representing multiple colors

At this point, we know that we can represent colors as points in a 3D space, and the Lab space looks good for this purpose, because we want Euclidean distances to correspond to perceptual judgements (more on this later).

But how do we represent a whole image, more likely than not composed of many colors?
My solution is to introduce a *palette* of colors, with the goal of approximately covering the color space.
Then, the color information of an image can be represented as a histogram over the palette.
In other words, for each color in the palette we find the percentage of pixels in the image that are *nearest* (in terms of Euclidean distance) to that color.

We can represent this information in a slightly different way, by showing the top colors present in the image in a type of palette image, with the area of a color in the palette image proportional to the prevalence of that color in the image.

![Example of an image, its color histogram, and its top colors.]()

![And here is the palette that was used]().

But now here is another image, which looks almost the same; however, its histogram has no overlap at all with the previous one.

![Example of an image, its color histogram, and its top colors.]()

We see the problem: although the colors are very close, they do not fall into exactly the same bins of the palette, and so the histograms have no overlap.

As we will discuss further, we want our histograms to have overlap 







---

We like Euclidean distance calculations.
They are quite fast due to their implementation as simple dot products (an operation for which there are very fast system-level libraries).


## Search

### Color histograms: representing multiple colors

### Forming the palette

- an image is represented as a color histogram
- a metric defines a similarity between two color histograms
  - can use different metrics, review which
- exact search takes O(N) time, and does comparisons in a relatively high-dimensional space

### Dimensionality Reduction

- PCA for dimensionality reduction

### Indexing search

- data structures for more efficient search
- cKDTree
- FLANN

### Similar images by color.

![Exact Euclidean](writeups/figures/matches_exact_euclidean_0.png)

### TODOs and Work Log

All planned and completed work is documented in the following public [Trello](https://trello.com/board/rayleigh/50d36a9e0f87f42952000276).

