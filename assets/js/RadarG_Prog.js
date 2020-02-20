var w = document.getElementById('chart_program').clientWidth/2;
var	h = w;

var colorscale = d3.scale.category10();

//Legend titles
var LegendOptions = ['2010','2013','2019'];

//Data
var d = [
		  [
			{axis:"Matlab",value:0.75},
			{axis:"LabView",value:0.90},
			{axis:"Cpp",value:0.95},
			{axis:"Java",value:0.4},
			{axis:"SQL",value:0.1},
			{axis:"HTML",value:0.2},
			{axis:"JavaScript",value:0.3},
			{axis:"Python",value:0.3},
			{axis:"CSS",value:0.2}
		  ],[
			{axis:"Matlab",value:0.90},
			{axis:"LabView",value:0.50},
			{axis:"Cpp",value:0.90},
			{axis:"Java",value:0.4},
			{axis:"SQL",value:0.4},
			{axis:"HTML",value:0.70},
			{axis:"JavaScript",value:0.75},
			{axis:"Python",value:0.6},
			{axis:"CSS",value:0.45}
		  ],[
			{axis:"Matlab",value:0.95},
			{axis:"LabView",value:0.70},
			{axis:"Cpp",value:0.80},
			{axis:"Java",value:0.70},
			{axis:"SQL",value:0.80},
			{axis:"HTML",value:0.50},
			{axis:"JavaScript",value:0.60},
			{axis:"Python",value:0.90},
			{axis:"CSS",value:0.55}
		  ]
		];
		
//Data
var d2 = [
		  [
			{axis:"Matlab",value:0},
			{axis:"LabView",value:0},
			{axis:"Cpp",value:0},
			{axis:"Java",value:0},
			{axis:"SQL",value:0},
			{axis:"HTML",value:0},
			{axis:"JavaScript",value:0},
			{axis:"Python",value:0},
			{axis:"CSS",value:0}
		  ],[
			{axis:"Matlab",value:0},
			{axis:"LabView",value:0},
			{axis:"Cpp",value:0},
			{axis:"Java",value:0},
			{axis:"SQL",value:0},
			{axis:"HTML",value:0},
			{axis:"JavaScript",value:0},
			{axis:"Python",value:0},
			{axis:"CSS",value:0}
		  ],[
			{axis:"Matlab",value:0},
			{axis:"LabView",value:0},
			{axis:"Cpp",value:0},
			{axis:"Java",value:0},
			{axis:"SQL",value:0},
			{axis:"HTML",value:0},
			{axis:"JavaScript",value:0},
			{axis:"Python",value:0},
			{axis:"CSS",value:0}
		  ]
		];

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
RadarChart.draw("#chart_program", d2, mycfg);
////////////////////////////////////////////
/////////// Initiate legend ////////////////
////////////////////////////////////////////
var svg = d3.select('#chart_title')
	.append('svg')
	.attr("width", 200)
	.attr("height", 200);

//Create the title for the legend
var text = svg.append("text")
	.attr("class", "title")
	.attr('transform', 'translate(90,0)') 
	.attr("x", 20)
	.attr("y", 20)
	.attr("font-size", "20px")
	.attr("fill", "#ffffff")
	.text("Skills");
		
//Initiate Legend	
var legend = svg.append("g")
	.attr("class", "legend")
	.attr("height", 100)
	.attr("width", 200)
	.attr('transform', 'translate(90,20)') 
	;
	//Create colour squares
	legend.selectAll('rect')
	  .data(LegendOptions)
	  .enter()
	  .append("rect")
	  .attr("x", 10)
	  .attr("y", function(d, i){ return i * 40 + 20;})
	  .attr("width", 20)
	  .attr("height", 20)
	  .style("fill", function(d, i){ return colorscale(i);})
	  ;
	//Create text next to squares
	legend.selectAll('text')
	  .data(LegendOptions)
	  .enter()
	  .append("text")
	  .attr("x", 35)
	  .attr("y", function(d, i){ return i * 40 + 35;})
	  .attr("font-size", "15px")
	  .attr("fill", "#ffffff")
	  .text(function(d) { return d; })
	  ;


//We will build a basic function to handle window resizing.
function resize_radar() {
	d3.select("chart_program").remove();
    mycfg.w = document.getElementById('chart_program').clientWidth/2;
    mycfg.h = w;
	RadarChart.draw("#chart_program", d, mycfg);
}

//Call our resize function if the window size is changed.
window.addEventListener("resize", resize_radar);

$(document).on('scroll', function() {
    if( $(this).scrollTop() >= $('#chart_program').position().top ){
		RadarChart.draw("#chart_program", d, mycfg);		
    }
});