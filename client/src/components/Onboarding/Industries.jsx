import React from "react";
import { Check, MoveRight } from 'lucide-react';
import { industries } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";

const Industries = () => {
    const { navigate } = useAppContext();

  return (
    <section className="container mx-auto mb-20 px-4 md:px-12 text-center space-y-12 text-black">
      {/* Header */}
      <div>
        <h3 className="text-sm font-bold uppercase tracking-wide text-special-red">
          Industries We Serve
        </h3>
        <h1 className="text-2xl md:text-5xl max-w-4xl mx-auto font-light mt-4 px-4 leading-snug">
          Our Brain Tumor Detection System powered by Deep Learning and AI
        </h1>
        <p className="mt-4 max-w-5xl mx-auto leading-snug font-light md:text-xl">
          has broad applications across multiple industries where medical
          imaging, diagnostics, and research play a vital role. The solution is
          designed to bring automation, accuracy, and efficiency to healthcare
          workflows.
        </p>
      </div>

      {/* Industries Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {industries.map((industry, index) => (
          <div
            key={index}
            className="bg-special-red/10 overflow-hidden hover:shadow transition border-t-4 border-special-red rounded-t"
          >
            <img
              loading="lazy"
              src={industry.img}
              alt={industry.title}
              className="w-full h-64 object-cover"
            />
            <div className="p-8">
              <h4 className="text-2xl">{industry.title}</h4>
              <p className="font-light md:text-lg mt-2">{industry.description}</p>
            </div>

            <div className="flex flex-col items-start px-8 mb-8 space-y-4">
              <h4 className="text-2xl">We help you:</h4>
              <ul className="space-y-4 font-light">
                {industry.helps.map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-start">
                    <div><Check className="text-special-red" size={20} /></div>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      {/* CTA Button */}
      <div onClick={() => { navigate('/industries'); scrollTo(0, 0) }}
        className="inline-flex items-center justify-center px-6 py-3 bg-special-red/10 text-black rounded-md hover:bg-special-red/15 shadow-sm transition group cursor-pointer"
      >
        More Industries
        <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
          <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
        </span>
      </div>
    </section>
  );
};

export default Industries;
