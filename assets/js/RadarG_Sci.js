var w = document.getElementById('chart_sci').clientWidth / 2;
var	h = w;

var colorscale = d3.scale.category10();

//Legend titles
var LegendOptions = ['2010','2013','2019'];

//Data
var d1 = [
		  [
			{axis:"SignalProcessing",value:0.90},
			{axis:"MachineLearning",value:0.40},
			{axis:"CompBiology",value:0.3},
			{axis:"DeepLearning",value:0.1},
			{axis:"Optimization",value:0.3},
			{axis:"CompScience",value:0.1}
		  ],[
			{axis:"SignalProcessing",value:0.75},
			{axis:"MachineLearning",value:0.85},
			{axis:"CompBiology",value:0.90},
			{axis:"DeepLearning",value:0.25},
			{axis:"Optimization",value:0.7},
			{axis:"CompScience",value:0.3}
		  ],[
			{axis:"SignalProcessing",value:0.68},
			{axis:"MachineLearning",value:0.75},
			{axis:"CompBiology",value:0.6},
			{axis:"DeepLearning",value:0.70},
			{axis:"Optimization",value:0.5},
			{axis:"CompScience",value:0.70}
		  ]
		];

var d2 = [
		  [
			{axis:"SignalProcessing",value:0},
			{axis:"MachineLearning",value:0},
			{axis:"CompBiology",value:0},
			{axis:"DeepLearning",value:0},
			{axis:"Optimization",value:0},
			{axis:"CompScience",value:0}
		  ],
		  [
			{axis:"SignalProcessing",value:0},
			{axis:"MachineLearning",value:0},
			{axis:"CompBiology",value:0},
			{axis:"DeepLearning",value:0},
			{axis:"Optimization",value:0},
			{axis:"CompScience",value:0}
		  ],
		  [
			{axis:"SignalProcessing",value:0},
			{axis:"MachineLearning",value:0},
			{axis:"CompBiology",value:0},
			{axis:"DeepLearning",value:0},
			{axis:"Optimization",value:0},
			{axis:"CompScience",value:0}
		  ]	  
		 ]

//Options for the Radar chart, other than default
var mycfg = {
  w: w,
  h: h,
  maxValue: 1,
  levels: 6,
  ExtraWidthX: 300
}

//Call function to draw the Radar chart
//Will expect that data is in %'s
RadarChart.draw("#chart_sci", d2, mycfg);
 
//We will build a basic function to handle window resizing.
function resize_radar() {
    mycfg.w = document.getElementById('chart_sci').clientWidth / 2;
    mycfg.h = w;
	RadarChart.draw("#chart_sci", d1, mycfg);
}

//Call our resize function if the window size is changed.
window.addEventListener("resize", resize_radar);

$(document).on('scroll', function() {
    if( $(this).scrollTop() >= $('#chart_sci').position().top ){
		RadarChart.draw("#chart_sci", d1, mycfg);		
    }
});