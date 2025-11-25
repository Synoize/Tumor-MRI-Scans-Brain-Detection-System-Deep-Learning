import React from "react";
import { MoveRight } from "lucide-react";
import { blogs } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";

const Blogs = () => {
  const { navigate } = useAppContext();

  return (
    <section className="container mx-auto mb-20 flex flex-col md:flex-row items-start justify-center">
      {/* Left Section */}
      <div className="md:w-1/2 w-full h-full p-8 md:p-12 space-y-12 bg-special-green text-black">
        <div className="space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-special-red">
            Blogs
          </h3>
          <h2 className="text-2xl md:text-5xl font-light leading-snug">
            Our project shares knowledge, progress, and insights
          </h2>
          <p className="font-light md:text-xl">
            About AI in medical imaging, deep learning advancements, and the
            impact of computer vision in healthcare. Each article aims to
            educate readers — students, developers, and healthcare
            professionals — about how artificial intelligence is transforming
            medical diagnostics.
          </p>
        </div>

        {/* Desktop Button */}
        <div onClick={() => { navigate('/blogs'); scrollTo(0, 0) }}
          className="hidden md:inline-flex items-center justify-center px-6 py-3 bg-white text-black rounded-md hover:bg-slate-50 shadow-sm transition mt-8 group cursor-pointer"
        >
          View All Blogs
          <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
            <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
          </span>
        </div>
      </div>

      {/* Right Section */}
      <div className="md:w-1/2 w-full p-8 md:p-12 bg-white rounded-r-2xl">
        <div className="space-y-4">
          {blogs.map((blog, index) => (
            <div
              key={index}
              className="p-4 border-l-4 border-special-red bg-slate-50 rounded-sm transform transition-transform duration-300 ease-in-out hover:scale-[1.02] group "
            >
              <div className="flex flex-col md:flex-row justify-between md:items-center gap-2">
                <h2 className="font-semibold">{blog.title}</h2>
                <p className="text-xs text-special-red text-nowrap">
                  {blog.date}
                </p>
              </div>
              <div className="flex justify-between">
                <p className="text-start font-light mt-2">{blog.description}</p>
                <div onClick={() => { navigate(blog.link); scrollTo(0, 0) }}
                  className="self-end mt-1 ml-1 inline-flex items-center justify-center w-10 h-full rounded-full text-sm cursor-pointer"
                >
                  <MoveRight size={24} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Mobile Button */}
        <div onClick={() => { navigate('/blogs'); scrollTo(0, 0) }}
          className="inline-flex items-center justify-center px-6 py-3 bg-special-red/10 md:bg-white text-black rounded-md hover:bg-special-red/15 md:hover:bg-slate-50 shadow-sm transition mt-8 md:hidden group cursor-pointer"
        >
          View All Blogs
          <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
            <MoveRight size={24} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
          </span>
        </div>
      </div>
    </section>
  );
};

export default Blogs;
