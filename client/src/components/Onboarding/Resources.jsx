import React from "react";
import { MoveRight } from "lucide-react";
import { assets } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";

const Resources = () => {
    const { navigate } = useAppContext();
  return (
    <section className="container mx-auto mb-20 md:mb-0 flex flex-col md:flex-row items-stretch justify-center bg-special-red/10 overflow-hidden shadow-sm">
      {/* Left Image */}
      <div className="md:w-1/2 w-full">
        <img
          loading="lazy"
          src={assets.resources}
          alt="Resources"
          className="w-full h-full object-cover"
        />
      </div>

      {/* Right Content */}
      <div className="md:w-1/2 w-full h-full p-8 md:p-14 space-y-10 text-black flex flex-col justify-center">
        <div className="space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-special-red">
            Resources
          </h3>
          <h2 className="text-2xl md:text-5xl font-light leading-snug">
            The development and implementation of this tool
          </h2>
          <p className="font-light md:text-xl text-gray-700">
            Relied on a combination of datasets, software tools, frameworks, and
            research publications. These resources collectively enabled
            efficient model training, image processing, and performance
            evaluation.
          </p>
        </div>

        <div onClick={() => { navigate('/resources'); scrollTo(0, 0) }}
          className="inline-flex items-center justify-center px-6 py-3 bg-white text-black rounded-md hover:bg-slate-100 shadow transition duration-300 group w-fit cursor-pointer"
        >
          More Resources
          <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
            <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
          </span>
        </div>
      </div>
    </section>
  );
};

export default Resources;
